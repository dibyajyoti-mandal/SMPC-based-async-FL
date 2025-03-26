"""flwr-app: A Flower / PyTorch app."""
import decimal

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import json
from flwr_app.task import Net, get_weights, load_data, set_weights, test, train
import numpy as np
import random
from typing import List, Dict, Tuple
from decimal import Decimal

# Import the registry at the top of your client_app.py
from flwr_app.global_client_registry import client_registry
from math import ceil


FIELD_SIZE = 10 ** 7  # Field size for secret sharing

def polynom(x, coefficients):
    """
    Generate a single point on the graph of a given polynomial.
    Args:
        x (int): The x-coordinate
        coefficients (List[int]): Coefficients of the polynomial
    Returns:
        int: The y-coordinate for the given x
    """
    point = 0
    for coefficient_index, coefficient_value in enumerate(coefficients[::-1]):
        point += x ** coefficient_index * coefficient_value
    return point

def coeff(t, secret):
    """
    Randomly generate coefficients for a polynomial.
    Args:
        t (int): Threshold for reconstruction
        secret (int): Secret value to be shared
    Returns:
        List[int]: Coefficients of the polynomial
    """
    coefficients = [random.randrange(0, FIELD_SIZE) for _ in range(t - 1)]
    coefficients.append(secret)
    return coefficients

def generate_shares(n, m, secret):
    coefficients = coeff(m, secret)
    shares = []
    used_x_values = set()  # Track used x values
    for i in range(1, n + 1):
        # Generate a unique x value
        while True:
            x = random.randrange(1, FIELD_SIZE)
            if x not in used_x_values:
                used_x_values.add(x)
                break
        shares.append((x, polynom(x, coefficients)))
    return shares

def reconstruct_secret(shares):
    sums = 0
    for j, share_j in enumerate(shares):
        xj, yj = share_j
        prod = Decimal(1)
        valid_share = True
        for i, share_i in enumerate(shares):
            xi, _ = share_i
            if i != j:
                try:
                    prod *= Decimal(Decimal(xi) / (xi - xj))
                except (ZeroDivisionError, decimal.DivisionByZero):
                    valid_share = False
                    break
        if valid_share:
            prod *= yj
            sums += Decimal(prod)
    return int(round(Decimal(sums), 0))

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, client_id, num_clients, buffer_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.client_id = client_id
        self.num_clients = num_clients
        self.buffer_id = buffer_id

        # Initialize my_shares as a dictionary of dictionaries for each client
        self.my_shares = {i: {} for i in range(num_clients)}
        self.received_shares = {}  # Initialize as a dictionary
        self.model_update = None
        self.temp_shares = {}  # Initialize as empty dictionary
        self.registry = client_registry

        print(" after server flow reaches inside flower client .....")

    def generate_model_update(self):
        """Generate model update and store it."""
        try:
            self.model_update = get_weights(self.net)
            print(f"Model has {len(self.model_update)} parameter groups")
            for i, param in enumerate(self.model_update):
                print(f"Parameter {i} shape: {param.shape}")
            return self.model_update
        except Exception as e:
            print(f"Error generating model update: {e}")
            return None

    def create_shares(self, threshold: int):
        """Create shares of the model update."""
        # Make sure model_update exists before creating shares
        if self.model_update is None:
            try:
                self.generate_model_update()
                if self.model_update is None:
                    print("Error: model_update is still None after generation")
                    return {}
                print(f"Model update contains {len(self.model_update)} layers")
            except Exception as e:
                print(f"Error generating model update: {e}")
                self.model_update = []
                return {}

        # Reset temp_shares as a dictionary
        self.temp_shares = {}

        # Initialize my_shares as a dictionary if it's not already
        if not isinstance(self.my_shares, dict):
            self.my_shares = {i: {} for i in range(self.num_clients)}

        # Handle the case where model_update is a list of tensors or numpy arrays
        if isinstance(self.model_update, list):
            # Process each layer separately
            for layer_idx, layer in enumerate(self.model_update):
                # Convert to numpy array if it's a tensor
                if isinstance(layer, torch.Tensor):
                    layer_data = layer.detach().cpu().numpy()
                else:
                    layer_data = np.array(layer)

                # Process the layer data
                self._process_layer(layer_idx, layer_data, threshold)
        else:
            # If it's a single array, process it directly
            self._process_layer(0, self.model_update, threshold)

        # Copy from my_shares to temp_shares in the correct format
        for recipient_id, shares in self.my_shares.items():
            if shares:  # Only include non-empty shares
                self.temp_shares[str(recipient_id)] = shares

        print(f"Shares have been generated for client {self.client_id} and stored in my_shares dictionary")
        print(f"temp_shares contains shares for {len(self.temp_shares)} recipients")

        # Add a copy of the values to the received_shares dictionary for this client
        if self.client_id not in self.received_shares:
            self.received_shares[self.client_id] = {}

        # Copy shares to received_shares for this client
        if str(self.client_id) in self.temp_shares:
            for position_index, share_data in self.temp_shares[str(self.client_id)].items():
                self.received_shares[self.client_id][position_index] = share_data

        return self.my_shares

    def _process_layer(self, layer_idx, layer_data, threshold):
        """Helper method to process a single layer of weights."""
        # Flatten multidimensional arrays for simplicity
        flat_data = layer_data.flatten()

        # Process each element in the flattened data
        for idx, value in enumerate(flat_data):
            int_value = int(value * 1000)  # Scale up to preserve precision
            shares = generate_shares(self.num_clients, threshold, int_value)

            # Create a unique index for this value
            position_index = f"layer{layer_idx}_idx{idx}"

            # Store shares for each client
            for recipient_id in range(self.num_clients):
                if recipient_id not in self.my_shares:
                    self.my_shares[recipient_id] = {}

                if position_index not in self.my_shares[recipient_id]:
                    self.my_shares[recipient_id][position_index] = {}

                self.my_shares[recipient_id][position_index] = {
                    'owner_id': self.client_id,
                    'share': shares[recipient_id],
                    'layer_idx': layer_idx,
                    'idx': idx,
                    'original_shape': layer_data.shape
                }

    def fit(self, parameters, config):
        # After creating shares
        print(f"inside the fit function for client number {self.client_id}")
        self.create_shares(threshold=config.get("threshold", 2))

        # Verify shares are properly included in received_shares
        if self.client_id not in self.received_shares:
            self.received_shares[self.client_id] = {}

        # Copy own shares to received_shares
        if str(self.client_id) in self.temp_shares:
            for position_index, share_data in self.temp_shares[str(self.client_id)].items():
                self.received_shares[self.client_id][position_index] = share_data

        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)

        # Convert shares to a serializable format
        serializable_shares = {}
        for recipient_id, shares in self.my_shares.items():
            if shares:  # Only include non-empty shares
                serializable_shares[str(recipient_id)] = {}
                for pos_idx, share_data in shares.items():
                    serializable_shares[str(recipient_id)][str(pos_idx)] = {
                        'owner_id': share_data['owner_id'],
                        'share': [int(share_data['share'][0]), int(share_data['share'][1])],
                        'layer_idx': int(share_data['layer_idx']),
                        'idx': int(share_data['idx']),
                        'original_shape': share_data['original_shape'].tolist() if hasattr(share_data['original_shape'], 'tolist') else share_data['original_shape']
                    }

        # Store the serializable shares for distribution
        self.temp_shares = serializable_shares

        print("now distributing the shares to all the clients ")
        self.registry.distribute_shares_to_all(self.client_id)

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "shares": json.dumps(serializable_shares)},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def print_serializable_shares_info(serializable_shares):
    print("\nPrinting Serializable Shares Information:")
    for recipient_id, positions in serializable_shares.items():  # Iterate over top-level keys
        print(
            f"\nRecipient ID: {recipient_id} (Type: {type(positions).__name__}, Number of Positions: {len(positions)})")
        for pos_idx, share_data in positions.items():  # Iterate over second-level keys
            print(f" Position Index: {pos_idx} (Type: {type(share_data).__name__}, Number of Keys: {len(share_data)})")
            for key, value in share_data.items():  # Iterate over dictionary inside
                value_type = type(value).__name__
                value_length = len(value) if hasattr(value, '__len__') else "N/A"
                print(f" Key: {key}, Type: {value_type}, Number of Values: {value_length}")

def client_fn(context: Context):
    # Load model and data
    print("Client function initialization started")

    # Get the singleton instance
    registry = client_registry

    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    threshold = context.run_config["threshold"]

    # Extract client ID and number of clients from the context
    client_id = partition_id  # Assuming partition_id is used as client_id
    num_clients = num_partitions  # Assuming num_partitions is the total number of clients
    buffer_id = partition_id  # Assuming buffer_id is the same as client_id

    # Create FlowerClient instance
    client = FlowerClient(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=local_epochs,
        client_id=client_id,
        num_clients=num_clients,
        buffer_id=buffer_id
    )

    print(f"adding client {client_id} in registry")
    registry.register_client(client_id, client)

    # Register the client in the global registry
    # In client_app.py where you're printing the registry
    print(f"registry initialised for the client no -{client_id}")
    print(f"Registry contents:\n{registry}")  # This will use the __str__ method

    # For more detailed information
    # registry.print_detailed()

    # Modify the fit method to use the registry for share distribution
    original_fit = client.fit

    # Store the original evaluate method
    original_evaluate = client.evaluate

    # Define the wrapped method that calls the original
    def wrapped_evaluate(parameters, config):
        result = original_evaluate(parameters, config)
        # After evaluation, print all received shares
        registry.print_all_received_shares()
        return result

    # Replace the evaluate method with our wrapped version
    client.evaluate = wrapped_evaluate

    # Return Client instance
    return client.to_client()

# Flower ClientApp
app = ClientApp(client_fn)
