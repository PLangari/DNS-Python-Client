import socket
import argparse
import struct


def create_dns_query(domain, query_type="A"):
    """
    Create a DNS query based on the given domain and type.
    """
    # Header section (simplified for the sake of demonstration)
    transaction_id = 0x1234
    flags = 0x0100  # standard query
    questions = 1
    answer_rrs = 0
    authority_rrs = 0
    additional_rrs = 0

    header = struct.pack(
        "!HHHHHH",
        transaction_id,
        flags,
        questions,
        answer_rrs,
        authority_rrs,
        additional_rrs,
    )

    # Question section
    q_name = b""
    for label in domain.split("."):
        q_name += struct.pack("B", len(label)) + label.encode()
    q_name += b"\x00"  # Null terminator

    if query_type == "A":
        q_type = 1  # A type
    elif query_type == "MX":
        q_type = 15  # MX type
    elif query_type == "NS":
        q_type = 2  # NS type

    q_class = 1  # Internet class
    question = q_name + struct.pack("!HH", q_type, q_class)

    return header + question


def dns_query(server, port, domain, query_type, timeout, max_retries):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)

        # Create the query
        query = create_dns_query(domain, query_type)

        retries = 0
        response = None
        while retries < max_retries:
            try:
                # Send the query
                sock.sendto(query, (server, port))

                # Await response
                response, _ = sock.recvfrom(512)
                break
            except socket.timeout:
                retries += 1

        if not response:
            print("ERROR\tMaximum number of retries exceeded")
            return

        # Parse and handle the response (to be implemented)
        # ...
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple DNS client.')
    parser.add_argument('-t', '--timeout', type=int, default=5, help='Timeout in seconds')
    parser.add_argument('-r', '--max-retries', type=int, default=3, help='Maximum retries')
    parser.add_argument('-p', '--port', type=int, default=53, help='Port of the DNS server')
    parser.add_argument('-mx', action='store_true', help='Query for MX record')
    parser.add_argument('-ns', action='store_true', help='Query for NS record')
    parser.add_argument('server', type=str, help='DNS server IP address')
    parser.add_argument('name', type=str, help='Domain name to query for')
    
    args = parser.parse_args()

    query_type = 'A'
    if args.mx:
        query_type = 'MX'
    elif args.ns:
        query_type = 'NS'

    print("timeout: ", args.timeout)
    print("max-retries: ", args.max_retries)
    print("port: ", args.port)
    print("query_type: ", query_type)
    print("server: ", args.server)
    print("name: ", args.name)
    

