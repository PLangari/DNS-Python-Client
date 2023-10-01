import struct

def parse_dns_response(response):
    # Unpacking header
    _, __, questions, answer_rrs, ____, _____ = struct.unpack(
        "!HHHHHH", response[:12]
    )

    if answer_rrs == 0:
        print("NOTFOUND")
        return
    
    offset = 12

    # Skipping the question section, just for simplicity
    for _ in range(questions):
        while response[offset] != 0:
            label_len = response[offset]
            offset += label_len + 1
        offset += 5  # termination label  (1 byte) + QTYPE (2 bytes) + QCLASS (2 bytes)

    print("***Answer Section(",answer_rrs, "records)***")
    for _ in range(answer_rrs):
        if offset + 12 > len(response):  # To ensure we have the full record header
            print("ERROR\tIncomplete record. Exiting.")
            break

        _, res_type, __, ttl, rd_length = struct.unpack(
            "!HHHLH", response[offset : offset + 12]
        )
        offset += 10

        if offset + rd_length > len(response):
            print("ERROR\tIncomplete record data. Exiting.")
            break

        rdata = response[offset : offset + rd_length]
        print(f"Type: {res_type}, TTL: {ttl}, Data: {rdata}")
        result = ""
        if res_type == 1:
            print("IP\t",".".join([str(int(b)) for b in rdata]),"\t",ttl,"\t")
        elif res_type == 2:
            print("NS\t",rdata.decode('utf-8'),"\t",ttl,"\t")
        elif res_type == 5:
            print("CNAME\t",rdata[2:].decode('utf-8'),"\t","".join([str(int(b)) for b in rdata[:2]]),"\t",ttl,"\t")
        elif res_type == 15:
            print("MX\t-",rdata.decode('utf-8'),"-\t",ttl,"\t")

        offset += rd_length