from flwr_app.global_client_registry import client_registry

def simulate_client_dropout(client_id):
    """
    Simulate a client dropout by removing the client from the registry or marking it inactive.

    Args:
        client_id (int): The ID of the client to simulate dropout for.
    """
    # Retrieve the client from the registry
    client = client_registry.get_client(client_id)
    
    if not client:
        print(f"Client {client_id} not found in the registry.")
        return
    
    # Option 1: Remove the client from the registry
    del client_registry.clients[client_id]
    print(f"Client {client_id} has been removed from the registry (simulating dropout).")
    
    # Option 2: Mark the client as inactive (if you prefer not to delete it)
    # You can add an "active" attribute to your client class and set it to False
    # Example:
    # client.active = False
    # print(f"Client {client_id} has been marked as inactive (simulating dropout).")

def simulate_dropout_after_shares_distributed():
    """
    Simulate dropout for a random subset of clients after shares are distributed.
    """
    all_clients = list(client_registry.get_all_clients().keys())
    
    if not all_clients:
        print("No clients available in the registry.")
        return
    
    # Randomly select clients to drop out
    dropout_count = max(1, len(all_clients) // 5)  # Drop out 20% of clients
    dropout_clients = random.sample(all_clients, dropout_count)
    
    for client_id in dropout_clients:
        simulate_client_dropout(client_id)

# Example usage:
simulate_dropout_after_shares_distributed()
