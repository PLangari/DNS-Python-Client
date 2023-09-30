import argparse

# Custom parser to print errors in args properly instead of throwing them
class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print("ERROR\t  Incorrect input syntax: ", message)
        exit(2)


def input_parser():
    # Using our custom parser
    parser = CustomArgumentParser(description="A simple DNS client.")

    parser.add_argument(
        "-t", "--timeout", type=int, default=5, help="Timeout in seconds"
    )
    parser.add_argument(
        "-r", "--max-retries", type=int, default=3, help="Maximum retries"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=53, help="Port of the DNS server"
    )
    parser.add_argument("-mx", action="store_true", help="Query for MX record")
    parser.add_argument("-ns", action="store_true", help="Query for NS record")
    parser.add_argument("@server", type=str, help="DNS server IP address")
    parser.add_argument("name", type=str, help="Domain name to query for")

    args = parser.parse_args()

    query_type = "A"
    if args.mx:
        query_type = "MX"
    elif args.ns:
        query_type = "NS"

    server_address = args.__dict__['@server'].replace("@", "")  # Remove @ if it exists

    port = args.port
    domain = args.name
    timeout = args.timeout
    max_retries = args.max_retries

    return server_address, port, domain, timeout, max_retries, query_type

