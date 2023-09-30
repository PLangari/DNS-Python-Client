from argsParser import input_parser
from dnsQuery import dns_query
from dnsResponse import parse_dns_response

# TODO: print accordingly IFF they are returned
# TODO: Determine if any additional information is needed to be printed
# TODO: do unexpected response error
# TODO: do the mtfnr report

def main():
    # Parse input
    server_address, port, domain, timeout, max_retries, query_type = input_parser()
    # Send query
    response = dns_query(server_address, port, domain, query_type, timeout, max_retries)
    # Parse response
    if response != "ERROR":
        parse_dns_response(response)

if __name__ == "__main__":
    main()
