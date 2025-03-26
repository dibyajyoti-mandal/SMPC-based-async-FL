class ClientRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClientRegistry, cls).__new__(cls)
            cls._instance.clients = {}  # Dictionary to store client instances
        return cls._instance

    def register_client(self, client_id, client):
        """Register a client in the registry."""
        self.clients[client_id] = client
        print(f"Client {client_id} registered in global registry")

    def get_client(self, client_id):
        """Get a client from the registry."""
        return self.clients.get(client_id)

    def get_all_clients(self):
        """Get all registered clients."""
        return self.clients

    def distribute_shares_to_all(self, sender_id):
        """Distribute shares from sender to all other clients."""
        print(f"distribute shares to all ................. called for the client no - {sender_id}")
        sender = self.clients.get(sender_id)
        if not sender:
            print(f"Sender client {sender_id} not found in registry")
            return

        # Debug: Print the structure of temp_shares
        print(f"Sender {sender_id} temp_shares structure:")
        print(f"Type of temp_shares: {type(sender.temp_shares)}")
        print(f"temp_shares contains {len(sender.temp_shares)} recipient entries")

        # Iterate through the dictionary properly
        for recipient_id_str, shares in sender.temp_shares.items():
            recipient_id = int(recipient_id_str)
            print(f" For recipient {recipient_id}: {len(shares)} shares")
            if shares:
                print(f" First few keys: {list(shares.keys())[:3]}")

        # For each recipient client
        for recipient_id, recipient_client in self.clients.items():
            # Skip self (we already have our own shares)
            if recipient_id == sender_id:
                continue

            recipient_id_str = str(recipient_id)
            print(f"Attempting to distribute shares to recipient {recipient_id}")

            # Check if there are shares for this recipient
            if recipient_id_str in sender.temp_shares and sender.temp_shares[recipient_id_str]:
                print(f"Found {len(sender.temp_shares[recipient_id_str])} shares for recipient {recipient_id}")

                # Check if recipient already has shares from this client
                if sender_id not in recipient_client.received_shares:
                    # Initialize the entry for this client in recipient's received_shares
                    recipient_client.received_shares[sender_id] = {}
                    print(f"Initialized received_shares[{sender_id}] for recipient {recipient_id}")

                # Copy all shares meant for this recipient
                shares_copied = 0
                for position_index, share_data in sender.temp_shares[recipient_id_str].items():
                    recipient_client.received_shares[sender_id][position_index] = share_data
                    shares_copied += 1
                print(f"Copied {shares_copied} shares to recipient {recipient_id}")
            else:
                print(f"No shares found for recipient {recipient_id}")

    # Optional: Add a more detailed representation method
    def print_detailed(self):
        """Print detailed information about the registry."""
        from pprint import pprint

        print(f"=== ClientRegistry with {len(self.clients)} clients ===")
        for client_id, client in self.clients.items():
            print(f"\n=== Client ID: {client_id} ===")

            print("\nReceived shares:")
            for sender_id, shares in client.received_shares.items():
                print(f"  From client {sender_id}:")
                for position, share_data in list(shares.items())[:3]:  # Print first 3 shares only
                    print(f"    Position {position}: {share_data}")
                if len(shares) > 100:
                    print(f"    ... and {len(shares) - 3} more positions")

            print("\nMy shares:")
            for recipient_id, shares in enumerate(client.my_shares):
                if shares:
                    print(f"  For client {recipient_id}:")
                    positions = list(shares.keys())[:3]  # Print first 3 positions only
                    for position in positions:
                        print(f"    Position {position}: {shares[position]}")
                    if len(shares) > 3:
                        print(f"    ... and {len(shares) - 3} more positions")

    def print_all_received_shares(self):
        """Print received shares information for all clients."""
        print("\n=== RECEIVED SHARES FOR ALL CLIENTS ===")
        for client_id, client in self.clients.items():
            print(f"\nClient {client_id} received shares from {len(client.received_shares)} clients:")
            for sender_id, shares in client.received_shares.items():
                print(f"  - From client {sender_id}: {len(shares)} shares")

            # Verify if client has received shares from all clients
            all_client_ids = set(self.clients.keys())
            received_from = set(client.received_shares.keys())
            missing_from = all_client_ids - received_from

            if missing_from:
                print(f"  ! Missing shares from clients: {missing_from}")
            else:
                print(f"  ✓ Received shares from all clients including itself")

    def __str__(self):
        """Return a string representation of the registry."""
        output = f"ClientRegistry with {len(self.clients)} clients:\n"
        for client_id, client in self.clients.items():
            output += f"\nClient ID: {client_id}\n"
            # Print received shares info
            output += f" Received shares from {len(client.received_shares)} clients\n"
            for sender_id, shares in client.received_shares.items():
                output += f" From client {sender_id}: {len(shares)} position shares\n"
            # Print temp_shares info
            output += f" My shares for other clients:\n"
            for recipient_id_str, shares in client.temp_shares.items():
                if shares:
                    output += f" For client {recipient_id_str}: {len(shares)} position shares\n"
        return output

# Create a single instance
client_registry = ClientRegistry()
    # Optionally, you can add:
__all__ = ['client_registry']


