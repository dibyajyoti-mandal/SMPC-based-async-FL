"""flwr-app: A Flower / PyTorch app."""
from flwr.common import Context, ndarrays_to_parameters, FitRes, FitIns
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from numpy.matlib import empty
from sympy.codegen.cnodes import sizeof

from flwr_app.task import Net, get_weights
import numpy as np
import random
import json
from typing import Dict, List
from decimal import Decimal
from flwr_app.global_client_registry import client_registry


class Buffer:
    def __init__(self, buffer_id: int, threshold: int):

        """
        Initialize a buffer for secret sharing.
        Args:
            buffer_id (int): Unique identifier for the buffer
            threshold (int): Minimum shares required for reconstruction
        """
        self.buffer_id = buffer_id
        self.threshold = threshold
        self.shares = {} # Shares from different senders
        self.total_shares_count = 0
        self.layer_shapes = {} # Store shapes of layers for reconstruction
        self.position_values = {} # Store shares by position for reconstruction

    def add_share(self, sender_id: int, position_index: str, share_data: Dict):
        """
        Add a share to the buffer.
        Args:
            sender_id (int): ID of the sending client
            position_index (str): Position index of the share
            share_data (Dict): Share data
        """
        if position_index not in self.position_values:
            self.position_values[position_index] = []

        # Store layer shape information if available
        if 'original_shape' in share_data:
            layer_idx = share_data.get('layer_idx', 0)
            if f"layer{layer_idx}" not in self.layer_shapes:
                self.layer_shapes[f"layer{layer_idx}"] = share_data['original_shape']

        self.position_values[position_index].append({
            'sender_id': sender_id,
            'share': share_data['share'],
            'owner_id': share_data['owner_id'],
            'layer_idx': share_data.get('layer_idx', 0),
            'idx': share_data.get('idx', 0)
        })
        self.total_shares_count += 1

    def is_ready_for_reconstruction(self):
        # Check if we have at least threshold number of shares for any position
        for position, shares in self.position_values.items():
            if len(shares) >= self.threshold:
                # Check for duplicate x values that would cause division by zero
                x_values = [share['share'][0] for share in shares]
                if len(x_values) == len(set(x_values)):  # All x values are unique
                    return True
        return False

    def reconstruct_update(self):
        """
        Reconstruct model update using all available shares.
        Returns:
            Dict or None: Reconstructed model updates by owner_id
        """
        reconstructed_values = {}
        # Group shares by owner_id
        shares_by_owner = {}
        for position, shares_list in self.position_values.items():
            for share_info in shares_list:
                owner_id = share_info['owner_id']
                layer_idx = share_info.get('layer_idx', 0)
                idx = share_info.get('idx', 0)
                if owner_id not in shares_by_owner:
                    shares_by_owner[owner_id] = {}
                position_key = f"layer{layer_idx}_idx{idx}"
                if position_key not in shares_by_owner[owner_id]:
                    shares_by_owner[owner_id][position_key] = []
                shares_by_owner[owner_id][position_key].append(share_info['share'])

        # Reconstruct values for each owner
        for owner_id, positions in shares_by_owner.items():
            reconstructed_values[owner_id] = {}
            for position, shares in positions.items():
                if len(shares) >= self.threshold:
                    try:
                        reconstructed_value = reconstruct_secret(shares)
                        reconstructed_values[owner_id][position] = reconstructed_value / 1000 # Scale back down
                    except Exception as e:
                        print(f"Error reconstructing value for owner {owner_id} at position {position}: {e}")

        # Convert to numpy arrays based on layer shapes
        final_reconstructions = {}
        for owner_id, positions in reconstructed_values.items():
            final_reconstructions[owner_id] = []
            # Group by layer
            layer_values = {}
            for position, value in positions.items():
                layer_idx = int(position.split('_')[0].replace('layer', ''))
                idx = int(position.split('_')[1].replace('idx', ''))
                if layer_idx not in layer_values:
                    layer_values[layer_idx] = {}
                layer_values[layer_idx][idx] = value

            # Reconstruct each layer
            for layer_idx in sorted(layer_values.keys()):
                layer_key = f"layer{layer_idx}"
                if layer_key in self.layer_shapes:
                    shape = self.layer_shapes[layer_key]
                    flat_size = np.prod(shape)
                    # Create flattened array
                    flat_array = np.zeros(flat_size)
                    for idx, value in layer_values[layer_idx].items():
                        if idx < flat_size:
                            flat_array[idx] = value
                    # Reshape to original shape
                    layer_array = flat_array.reshape(shape)
                    final_reconstructions[owner_id].append(layer_array)
                else:
                    print(f"Warning: Shape information missing for layer {layer_idx}")
        return final_reconstructions

def reconstruct_secret(shares):
    """
    Reconstruct secret using Lagrange interpolation.
    Args:
        shares (List[Tuple[int, int]]): Shares to reconstruct secret
    Returns:
        int: Reconstructed secret
    """
    sums = 0
    for j, share_j in enumerate(shares):
        xj, yj = share_j
        prod = Decimal(1)
        for i, share_i in enumerate(shares):
            xi, _ = share_i
            if i != j:
                prod *= Decimal(Decimal(xi) / (xi - xj))
        prod *= yj
        sums += Decimal(prod)
    return int(round(Decimal(sums), 0))

class FederatedServer:
    def __init__(self , clients, threshold , net):
        """
        Initialize the federated server.
        Args:
            clients (List): List of client objects
            threshold (int): Minimum shares required for reconstruction
        """

        self.clients = clients
        self.threshold = threshold
        self.buffers = []
        self.net = net
        self.setup_buffers()

    def setup_buffers(self):
        """Initialize buffers for each client."""
        print(f"Setting up {len(self.clients)} buffers with threshold {self.threshold}")
        self.buffers = []  # Clear existing buffers
        for i in range(len(self.clients)):
            self.buffers.append(Buffer(buffer_id=i, threshold=self.threshold))
        print(f"Created {len(self.buffers)} buffers")

    # ... (rest of the class remains the same)

    def process_client_shares(self, client_shares):
        print("\nForwarding shares to buffers...")
        for client_id_str, client_data in client_shares.items():
            client_id = int(client_id_str)
            print(f"Processing shares from client {client_id}")

            # Process all shares from this client
            for recipient_id_str, shares in client_data.items():
                for position_index, share_data in shares.items():
                    # Forward to the buffer matching the owner_id
                    owner_id = share_data.get('owner_id')
                    if owner_id < len(self.buffers):
                        self.buffers[owner_id].add_share(client_id, position_index, share_data)

    def reconstruct_updates(self):
        print("\nReconstructing model updates...")
        reconstructed_updates = {}
        for buffer in self.buffers:
            print(f"Checking buffer {buffer.buffer_id}")
            print(f"Total shares in buffer: {buffer.total_shares_count}")
            if buffer.is_ready_for_reconstruction():
                print(f"Reconstructing update for buffer {buffer.buffer_id}")
                reconstructed = buffer.reconstruct_update()
                if reconstructed:
                    for owner_id, model in reconstructed.items():
                        print(f"Reconstructed model for owner {owner_id}")
                        reconstructed_updates[owner_id] = model
                        print(f"reconstruction has  has {len(model)} parameter groups")
                        for i, param in enumerate(model):
                            print(f"Parameter {i} shape: {param.shape}")

                else:
                    print(f"Failed to reconstruct update for buffer {buffer.buffer_id}")
            else:
                print(f"Buffer {buffer.buffer_id} not ready for reconstruction")
        return reconstructed_updates


def server_fn(context: Context):

    # Read from config
    # this is loaded from pyproject.toml me tool.flwr.app.config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    threshold = context.run_config["threshold"]
    print(" threshold in server_fn is : ", threshold)

    # Get number of clients from context
    #this is loaded from pyproject.toml me tool.flwr.federations.local-simulation
    num_clients = context.run_config["num-partitions"]

    #setup a global client registry
    registry = client_registry
    # generate the code for calling global_client_registry.py and initialising global client registry for each of the client

    # Initialize model parameters
    net = Net()
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    print(f"Model has {len(ndarrays)} parameter groups")
    for i, param in enumerate(ndarrays):
        print(f"Parameter {i} shape: {param.shape}")

    # Initialize federated server with buffers
    server = FederatedServer(clients=list(range(num_clients)), threshold=threshold , net=net)

    # Custom FedAvg strategy that handles reconstruction
    class SecureAggStrategy(FedAvg):
        def __init__(self, federated_server, client_registry, **kwargs):
            super().__init__(**kwargs)
            self.federated_server = federated_server
            self.client_registry = client_registry
            self.client_shares = {}

        def aggregate_fit(self, server_round, results, failures):
            print(f"Aggregating fit results for round {server_round}")
            print(f"Received {len(results)} results and {len(failures)} failures")

            for client_proxy, fit_res in results:
                client_id = client_proxy.cid
                print(f"Processing result from client {client_id}")
                if fit_res.metrics and "shares" in fit_res.metrics:
                    try:
                        shares_str = fit_res.metrics["shares"]
                        self.client_shares[str(client_id)] = json.loads(shares_str)
                    except Exception as e:
                        print(f"Error parsing shares for client {client_id}: {e}")

            # Check that shares have the expected structure and content
            for client_id, shares in self.client_shares.items():
                print(f"Client {client_id} provided shares for {len(shares)} recipients")

                # Check if any shares were provided
                if not shares:
                    print(f"Warning: Client {client_id} did not provide any shares")
                    continue

                # Check the first recipient's shares to see what layers are included
                first_recipient = next(iter(shares))
                if not shares[first_recipient]:
                    print(f"Warning: Client {client_id}'s shares for recipient {first_recipient} are empty")
                    continue

                # Analyze the layer structure by looking at position keys
                layer_indices = set()
                for position_key in shares[first_recipient].keys():
                    if position_key.startswith("layer"):
                        layer_idx = int(position_key.split("_")[0].replace("layer", ""))
                        layer_indices.add(layer_idx)

                print(f"Client {client_id} provided shares for layers: {sorted(layer_indices)}")

                # Check if we're missing any expected layers
                expected_layers = set(range(len(get_weights(self.federated_server.net))))
                missing_layers = expected_layers - layer_indices
                if missing_layers:
                    print(f"Warning: Client {client_id} is missing shares for layers: {sorted(missing_layers)}")
                else:
                    print(f"Client {client_id} provided shares for all {len(expected_layers)} expected layers")

            # Process shares and reconstruct updates
            if self.client_shares:
                self.federated_server.process_client_shares(self.client_shares)
                reconstructed_updates = self.federated_server.reconstruct_updates()

                if reconstructed_updates:
                    print("\nReconstructed updates: hehe ")
                    print(f"Using {len(reconstructed_updates)} reconstructed updates for aggregation")
                    secure_results = []

                    # print(f"printing results  {results[0]}" )
                    print(f"printing failures  {failures}" )
                    print("ab saala results ke andar enumerate kyu nahi ho rha !! ")
                    # After reconstructing updates
                    print(f"reconstructed_updates keys: {list(reconstructed_updates.keys())}")
                    print(f"client IDs in results: are {[int(client_proxy.cid) for client_proxy, _ in results]}")
                    print("---------------------------------------------")
                    for client_idx, (client_proxy, fit_res) in enumerate(results):
                        # print(f"Processing result from client {client_idx}")
                        client_id = int(client_proxy.cid)
                        print(f"client_id is {client_id}")
                        # print("reconstructed_updates : --------- ", reconstructed_updates)
                        if client_idx in reconstructed_updates:
                            print(f" bkl flow reached here for  {client_idx} ")
                            new_fit_res = FitRes(
                                parameters=ndarrays_to_parameters(reconstructed_updates[client_idx]),
                                num_examples=fit_res.num_examples,
                                metrics=fit_res.metrics,
                                status=fit_res.status
                            )
                            print(f"new_fit_res is also returned for client {client_idx}")
                            secure_results.append((client_proxy, new_fit_res))
                            print(f"Reconstructed update for client {client_id}:")
                            for i, layer in enumerate(reconstructed_updates[client_idx]):
                                print(f" Layer {i} shape: {layer.shape}, mean: {np.mean(layer):.6f}")

                    if secure_results:
                        print("Calling super().aggregate_fit() with reconstructed updates")
                        return super().aggregate_fit(server_round, secure_results, failures)

            print("falling back to super().aggregate_fit() without reconstructed updates")
            return super().aggregate_fit(server_round, results, failures)

        def configure_fit(self, server_round, parameters, client_manager):
            """Configure the next round of training."""
            # Get base configuration from parent
            client_instructions = super().configure_fit(server_round, parameters, client_manager)

            # Create a new list of client instructions with modified configs
            new_client_instructions = []

            for client, fit_ins in client_instructions:
                # Create a new config dictionary with the threshold
                config = dict(fit_ins.config)  # Convert to dict first
                config["threshold"] = self.federated_server.threshold

                # Create a new FitIns with the updated config
                new_fit_ins = FitIns(parameters=fit_ins.parameters, config=config)

                # Add to the new list
                new_client_instructions.append((client, new_fit_ins))

            return new_client_instructions


    # Define strategy with our custom strategy
    strategy = SecureAggStrategy(
        federated_server=server,
        client_registry=registry,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=5,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
