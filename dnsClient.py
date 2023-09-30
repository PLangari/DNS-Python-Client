import socket
import argparse
import struct

# TODO: return something, so remove prints
# TODO: take care of DEFAULT values to print
# TODO: determine how to determine if CNAME, MX, NS are returned or not
# TODO: print accordingly IFF they are returned
# TODO: Determine if any additional information is needed to be printed
# TODO: show NOTFOUND if no records are returned
# TODO: show errors if errors are found
# TODO: do the mtfnr report
def parse_dns_response(response):
    # Unpacking header
    transaction_id, flags, questions, answer_rrs, ns_count, ar_count = struct.unpack(
        "!HHHHHH", response[:12]
    )

    print(f"Transaction ID: {transaction_id}")
    print(f"Flags: {flags}")
    print(f"Questions: {questions}")
    print(f"Answer RRs: {answer_rrs}")
    print(f"Authority RRs: {ns_count}")
    print(f"Additional RRs: {ar_count}")

    offset = 12

    # Skipping the question section, just for simplicity
    for _ in range(questions):
        while response[offset] != 0:
            label_len = response[offset]
            offset += label_len + 1
        offset += 5  # QTYPE + QCLASS

    print("Answers:")
    for _ in range(answer_rrs):
        if offset + 12 > len(response):  # To ensure we have the full record header
            print("Incomplete record. Exiting.")
            break

        name_ptr, res_type, res_class, ttl, rd_length = struct.unpack(
            "!HHHLH", response[offset : offset + 12]
        )
        offset += 10

        if offset + rd_length > len(response):
            print("Incomplete record data. Exiting.")
            break

        rdata = response[offset : offset + rd_length]
        print(f"Type: {res_type}, TTL: {ttl}, Data: {rdata}")
        offset += rd_length


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
    q_name += b"\x00"  # Terminating byte

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
        # TODO: start timer here
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

        parse_dns_response(response)


def input_parser():
    parser = argparse.ArgumentParser(description="A simple DNS client.")
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
    parser.add_argument("server", type=str, help="DNS server IP address")
    parser.add_argument("name", type=str, help="Domain name to query for")

    args = parser.parse_args()

    query_type = "A"
    if args.mx:
        query_type = "MX"
    elif args.ns:
        query_type = "NS"

    server_address = args.server.replace("@", "")  # Remove @ if it exists
    port = args.port
    domain = args.name
    timeout = args.timeout
    max_retries = args.max_retries

    return server_address, port, domain, timeout, max_retries, query_type


def main():
    server_address, port, domain, timeout, max_retries, query_type = input_parser()

    dns_query(server_address, port, domain, query_type, timeout, max_retries)


if __name__ == "__main__":
    main()
