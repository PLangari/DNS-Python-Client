import struct

def parse_dns_response(response):
    # Unpacking header
    _, flags, questions, answer_rrs, auth_rrs, additional_rrs = struct.unpack(
        "!HHHHHH", response[:12]
    )
    
    auth = process_flags(flags)
    
    if auth.startswith("ERROR"):
        print(auth)
        return
    elif auth == "NOTFOUND":
        print(auth)
        return 
    
    if answer_rrs == 0:
        print("NOTFOUND")
        return
    
    offset = 12

    # Skipping the question section
    for _ in range(questions):
        while response[offset] != 0:
            label_len = response[offset]
            offset += label_len + 1
        offset += 5  # termination label  (1 byte) + QTYPE (2 bytes) + QCLASS (2 bytes)

    # Parse answer section
    print("***Answer Section(",answer_rrs, "records)***")
    for _ in range(answer_rrs):
        if offset + 12 > len(response):  # To ensure we have the full record header
            print("ERROR\tIncomplete answer record. Exiting.")
            return

        while True:
            if response[offset] == 0:
                offset+=1
                break
            if response[offset] & 0xC0 == 0xC0:
                offset += 2
                break
            offset += 1

        res_type = int.from_bytes(response[offset: offset+2], byteorder='big')
        offset += 2

        # Verify class bits
        classField = int.from_bytes(response[offset: offset+2], byteorder='big')
        if classField != 0x0001:
            print("ERROR\tIncorrect class field in the answer section. Exiting")
            return
        offset += 2

        ttl = int.from_bytes(response[offset: offset+4], byteorder='big')
        offset += 4

        rd_length = int.from_bytes(response[offset:offset+2], byteorder='big')
        offset += 2

        if offset + rd_length > len(response):
            print("ERROR\tIncomplete answer record data. Exiting.")
            return
        rdata = response[offset : offset + rd_length] 
        print(response[offset:])
        print("RDATA:", rdata)
        if res_type == 1:
            print("IP\t",".".join([str(int(b)) for b in rdata]),"\t",ttl,"\t",auth)
        elif res_type == 2:
            alias = parse_answer_data(response, offset)
            print("NS\t",alias,"\t",ttl,"\t",auth)
        elif res_type == 5:
            alias = parse_answer_data(response, offset)
            print("CNAME\t",alias,"\t",ttl,"\t",auth)
        elif res_type == 15:
            pref = int.from_bytes(response[offset:offset+2], "big")
            offset +=2 
            alias = parse_answer_data(response, offset)
            print("MX\t",alias,"\t",pref,"\t",ttl,"\t",auth)

        offset += rd_length - 2
    
    # Skip over authority section
    for _ in range(auth_rrs):
        if offset + 12 > len(response):  # To ensure we have the full record header
            print("ERROR\tIncomplete authority record. Exiting.")
            return

        while True:
            if response[offset] == 0:
                offset+=1
                break
            if response[offset] & 0xC0 == 0xC0:
                offset += 2
                break
            offset += 1
        
        offset += 8
    
        rd_length = int.from_bytes(response[offset:offset+2], byteorder='big')
        offset += 2

        if offset + rd_length > len(response):
            print("ERROR\tIncomplete authority record data. Exiting.")
            break
        rdata = response[offset : offset + rd_length] 
        

        offset += rd_length

    # Parse additional section
    if (additional_rrs > 0):
        print("***Additional Section(",additional_rrs, "records)***")
    for _ in range(additional_rrs):
        if offset + 12 > len(response):  # To ensure we have the full record header
            print("ERROR\tIncomplete additional record. Exiting.")
            break

        while True:
            if response[offset] == 0:
                offset += 1
                break
            if response[offset] & 0xC0 == 0xC0:
                offset += 2
                break
            offset += 1

        res_type = int.from_bytes(response[offset: offset+2], byteorder='big')
        offset += 2

        # Verify class bits
        classField = int.from_bytes(response[offset: offset+2], byteorder='big')
        if classField != 0x0001:
            print("ERROR\tIncorrect class field in the additional section. Exiting")
            return
        offset += 2

        ttl = int.from_bytes(response[offset: offset+4], byteorder='big')
        offset += 4

        rd_length = int.from_bytes(response[offset:offset+2], byteorder='big')
        offset += 2

        if offset + rd_length > len(response):
            print("ERROR\tIncomplete additonal record data. Exiting.")
            return
        
        rdata = response[offset : offset + rd_length] 
        
        if res_type == 1:
            print("IP\t",".".join([str(int(b)) for b in rdata]),"\t",ttl,"\t",auth)
        elif res_type == 2:
            alias = parse_answer_data(response, offset)
            print("NS\t",alias,"\t",ttl,"\t",auth)
        elif res_type == 5:
            alias = parse_answer_data(response, offset)
            print("CNAME\t",alias,"\t",ttl,"\t",auth)
        elif res_type == 15:
            pref = int.from_bytes(response[offset:offset+2], "big")
            offset+=2
            alias = parse_answer_data(response, offset)
            print("MX\t",alias,"\t",pref,"\t",ttl,"\t",auth)

        offset += rd_length
    

def parse_answer_data(response, offset):
    data = []
    ptr = offset

    while True:
        if response[ptr] == 0:
            break
        elif response[ptr] & 0xC0 == 0xC0:
            ptr = int(response[ptr+1])
        else:
            label_len = response[ptr]
            label = response[ptr + 1 : ptr + 1 + label_len]
            data.append(label.decode("utf-8"))
            ptr += label_len + 1
    
    return ".".join(data)

def process_flags(flags):
    # Define constants for flag positions
    QR_MASK = 0x80  # 10000000
    OPCODE_MASK = 0x78  # 01111000
    AA_MASK = 0x04  # 00000100
    TC_MASK = 0x02  # 00000010
    RD_MASK = 0x01  # 00000001

    flags_byte = flags

    # AA (Authoritative Answer)
    is_auth = flags_byte & AA_MASK != 0

    # RCODE (Response Code)
    rcode = flags_byte & 0x0F
    response_codes = {
        0: "No error condition",
        1: "Format error",
        2: "Server failure",
        3: "Name error",
        4: "Not implemented",
        5: "Refused",
    }
    
    if rcode == 0:
        if is_auth:
            return "auth"
        else:
            return "nonauth"
    elif rcode == 3:
        return "NOTFOUND"
    elif rcode in response_codes:
        return "ERROR\t{}".format(response_codes[rcode])
    else:
        return "ERROR\tUnnkown Error"

def skip_auth(response, offset, auth_rrs):
    for _ in range(auth_rrs):
        if offset + 12 > len(response):  # To ensure we have the full record header
            print("ERROR\tIncomplete record. Exiting.")
            break

        while True:
            if response[offset] == 0:
                offset+=1
                break
            if response[offset] & 0xC0 == 0xC0:
                offset += 2
                break
            offset += 1
        
        offset += 8

        rd_length = int.from_bytes(response[offset:offset+2], byteorder='big')
        offset += 2

        if offset + rd_length > len(response):
            print("ERROR\tIncomplete record data. Exiting.")
            break
        rdata = response[offset : offset + rd_length] 
        

        offset += rd_length