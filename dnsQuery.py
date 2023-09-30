import socket
import struct
import time

def dns_query(server, port, domain, query_type, timeout, max_retries):
    # Printing default values
    print("DnsClient sending for", domain)
    print("Server:", server)
    print("Request type:", query_type)
 
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)

        # Creating the query
        query = create_dns_query(domain, query_type)

        retries = 0
        response = None
        start = 0
        end = 0
        runtime = None
        # Start timer right before sending serie of queries
        # Send queries until we receive a response or max number of retries reached
        while retries < max_retries:
            try:
                # Send the query
                start = time.time() 
                sock.sendto(query, (server, port))
                # Await response
                response, _ = sock.recvfrom(1024)
                end = time.time()
                runtime = end - start
                break
            except socket.timeout:
                retries += 1
            except Exception as e:
                print("ERROR\t", e)
                return "ERROR"

        if not response:
            print("ERROR\tMaximum number of retries", max_retries," exceeded")
            return "ERROR"

        print("Response resceived after", runtime, "seconds (",retries, "retries)")
        return response

def create_dns_query(domain, query_type="A"):
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

