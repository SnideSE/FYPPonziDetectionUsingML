
import json
import random
import requests
import numpy as np
import pandas as pd
from web3 import Web3
from datetime import datetime
from pyevmasm import disassemble_all

features = [
            'address',
            'balance',
            "num_instructions",
            "num_unique_instructions",
            "num_jumps",
            "num_cond_jumps",
            "num_function_calls",
            "num_external_calls",
            "num_internal_calls",
            "num_push_instructions",
            "num_pop_instructions",
            "num_stack_operations",
            "num_sload_sstore",
            "num_arith_logic",
            "stack_depth",
            "code_size",
            "calling_instructions_ratio", 
            "timestamp_dependency", 
            "cyclomatic_complexity",
            "avg_gas_cost_per_instr",
            "gas_cost",
            'TotalGet',
            'TotalSend',
            'max_send',
            'avg_fee',
            'lifetime',
            'addr_get_profit',
            'gini_value',
            "bytecode"
            "opcodes",
            "instruction_counts",
]

w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/504412345678")) #Use own Infura API Key

API_KEY1 = 'EK-tycKg-123' #Use your own Ethplorer API Key
API_KEY2 = 'EK-496EB-123' #Use your own Ethplorer API Key

def get_address_info(address, api_key):
    url = f'https://api.ethplorer.io/getAddressInfo/{address}?apiKey={api_key}&showETHTotals=true'
    response = requests.get(url)
    return json.loads(response.text)

def get_address_transactions(address, api_key, limit=100):
    url = f'https://api.ethplorer.io/getAddressTransactions/{address}?apiKey={api_key}&showZeroValues=true'
    response = requests.get(url)
    return json.loads(response.text)

def get_tx_info(tx_hash, api_key):
    url = f'https://api.ethplorer.io/getTxInfo/{tx_hash}?apiKey={api_key}'
    response = requests.get(url)
    return json.loads(response.text)

def gini_co(array):
    if array is None or np.all(array == 0):
        return 0
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def get_lifetime(trans_timestamps):
    earliest_timestamp = min(trans_timestamps)
    latest_timestamp = max(trans_timestamps)
    lifetime = latest_timestamp - earliest_timestamp
    return lifetime

def get_addr_get_profit(transactions_data):
    recipients = []
    for tx in transactions_data:
        recipients.append(tx['to'])
        unique_recipients = set(recipients)
    return len(unique_recipients)

def get_avg_fee(transactions_data):
    gas_used = []
    for tx in transactions_data:
        gas_used.append(tx['gasUsed'])
    total_gas_used = sum(gas_used)
    return total_gas_used / len(gas_used)

def get_max_send(transactions_data):
    values = []
    for tx in transactions_data:
        values.append(tx['value'])
    return max(values)

def get_gini_value(transactions_data):
    values = []
    for tx in transactions_data:
        values.append(tx['value'])
    return values

def calculate_gas_cost(opcodes):
    gas_map = {
        "STOP": 0, "ADD": 3, "MUL": 5, "SUB": 3, "DIV": 5, "SDIV": 5, "MOD": 5, "SMOD": 5, "ADDMOD": 8, "MULMOD": 8, "EXP": 10, "SIGNEXTEND": 5,
        "LT": 3, "GT": 3, "SLT": 3, "SGT": 3, "EQ": 3, "ISZERO": 3, "AND": 3, "OR": 3, "XOR": 3, "NOT": 3, "BYTE": 3, "SHL": 3, "SHR": 3, "SAR": 3,
        "SHA3": 30, "ADDRESS": 2, "BALANCE": 400, "ORIGIN": 2, "CALLER": 2, "CALLVALUE": 2, "CALLDATALOAD": 3, "CALLDATASIZE": 2, "CALLDATACOPY": 3, "CODESIZE": 2, "CODECOPY": 3, "GASPRICE": 2, "EXTCODESIZE": 700, "EXTCODECOPY": 700, "RETURNDATASIZE": 2, "RETURNDATACOPY": 3, "EXTCODEHASH": 400, "BLOCKHASH": 20, "COINBASE": 2, "TIMESTAMP": 2, "NUMBER": 2, "DIFFICULTY": 2, "GASLIMIT": 2,
        "POP": 2, "MLOAD": 3, "MSTORE": 3, "MSTORE8": 3, "SLOAD": 200, "SSTORE": 20000, "JUMP": 8, "JUMPI": 10, "PC": 2, "MSIZE": 2, "GAS": 2, "JUMPDEST": 1, "PUSH": 3, "DUP": 3, "SWAP": 3,
        "LOG": 375, "CREATE": 32000, "CALL": 700, "CALLCODE": 700, "RETURN": 0, "DELEGATECALL": 700, "CREATE2": 32000, "STATICCALL": 700, "REVERT": 0, "INVALID": 0, "SELFDESTRUCT": 5000,
    }
    total_gas = 0
    for instruction in opcodes:
        total_gas += gas_map.get(instruction.mnemonic, 0)
    return total_gas

def get_calling_instructions_ratio(opcodes):
    num_calling_instructions = len([instr for instr in opcodes if instr.mnemonic in ["CALL", "CALLCODE", "DELEGATECALL"]])
    total_instructions = len(opcodes)
    if total_instructions == 0:
        return 0
    else:
        return num_calling_instructions / total_instructions

def has_timestamp_dependency(opcodes):
# Check if the contract depends on the timestamp
    for instr in opcodes:
        if instr.mnemonic == "TIMESTAMP":
            return True
    return False

def read_addresses_from_csv(file_name):
    addresses = []
    with open(file_name, 'r') as f:
        for line in f:
            addresses.append(line.strip())
    return addresses

def write_data_to_csv(data, file_name):
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)

####################################################################    
####################################################################
def extract_data(addresses, api_key):
    features = []
    counter = 0
    for address in addresses:
        try:
            print(f"{(counter := counter + 1)}:{address}")
            address = w3.to_checksum_address(address)
            address_info = get_address_info(address, api_key)
            address_transactions = get_address_transactions(address, api_key)

            # Define the block number
            try:
                block_number = w3.eth.block_number
            except ValueError:
                continue

            # Get the bytecode of the contract
            try:
                bytecode = w3.eth.get_code(address, block_number)
            except ValueError:
                print("No code at address %s for block %d" % (address, block_number))
                continue 

            # Convert the bytecode to a hex string and remove the "0x" prefix
            bytecode_hex = bytecode.hex()[2:]

            # Disassemble the bytecode into opcodes
            opcodes = [instruction for instruction in disassemble_all(bytecode)]

            # Extract code features from the opcodes
            num_instructions = len(opcodes)
            unique_instructions = set([instruction.mnemonic for instruction in opcodes])
            num_unique_instructions = len(unique_instructions)
            num_jumps = len([instruction for instruction in opcodes if instruction.mnemonic in ["JUMP", "JUMPI", "JUMPDEST"]])
            num_cond_jumps = len([instruction for instruction in opcodes if instruction.mnemonic == "JUMPI"])
            num_function_calls = len([instruction for instruction in opcodes if instruction.mnemonic in ["CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"]])
            num_external_calls = len([instruction for instruction in opcodes if instruction.mnemonic in ["CALL", "CALLCODE"]])
            num_internal_calls = len([instruction for instruction in opcodes if instruction.mnemonic in ["DELEGATECALL", "STATICCALL"]])
            num_push_instructions = len([instruction for instruction in opcodes if instruction.mnemonic.startswith("PUSH")])
            num_pop_instructions = len([instruction for instruction in opcodes if instruction.mnemonic in ["POP", "POP1", "SWAP1", "SWAP2", "SWAP3", "SWAP4", "SWAP5", "SWAP6", "SWAP7", "SWAP8", "SWAP9", "SWAP10", "SWAP11", "SWAP12", "SWAP13", "SWAP14", "SWAP15", "SWAP16"]])
            num_stack_operations = num_push_instructions + num_pop_instructions
            num_sload_sstore = len([instruction for instruction in opcodes if instruction.mnemonic in ["SLOAD", "SSTORE"]])
            arith_logic_instructions = ["ADD", "MUL", "SUB", "DIV", "MOD", "EXP", "AND", "OR", "XOR", "NOT", "LT", "GT", "SLT", "SGT", "EQ"]
            num_arith_logic = len([instruction for instruction in opcodes if instruction.mnemonic in arith_logic_instructions])
            cyclomatic_complexity = num_instructions - num_jumps + 2 * num_cond_jumps
            calling_instructions_ratio = get_calling_instructions_ratio(opcodes)  
            timestamp_dependency = has_timestamp_dependency(opcodes)  
            gas_cost = calculate_gas_cost(opcodes)
            code_size = len(bytecode)
            if num_instructions == 0:
                avg_gas_cost_per_instr = 0
            else:
                avg_gas_cost_per_instr = gas_cost / num_instructions

            instruction_counts = {}
            for instruction in opcodes:
                mnemonic = instruction.mnemonic
                if mnemonic not in instruction_counts:
                    instruction_counts[mnemonic] = 0
                instruction_counts[mnemonic] += 1

            stack_depth = 0
            max_stack_depth = 0
            for instruction in opcodes:
                if instruction.mnemonic.startswith("PUSH"):
                    stack_depth += 1
                    max_stack_depth = max(max_stack_depth, stack_depth)
                elif instruction.mnemonic.startswith("POP"):
                    stack_depth -= 1
                elif instruction.mnemonic.startswith("SWAP"):
                    pass
                else:
                    max_stack_depth = max(max_stack_depth, stack_depth)
                    
            transactions_data = []
            trans_timestamps = []
            value = []
            for tx in address_transactions:
                tx_info = get_tx_info(tx['hash'], api_key)
                transactions_data.append({
                    'timestamp': tx_info['timestamp'],
                    'from': tx_info['from'],
                    'to': tx_info['to'],
                    'value': tx_info['value'],
                    'gasUsed': tx_info['gasUsed']
                })
                trans_timestamps.append(tx_info['timestamp'])   
                value.append(tx_info['value'])

            balance = address_info['ETH']['balance']
            totalIn = address_info['ETH']['totalIn']
            totalOut = address_info['ETH']['totalOut']
            countTxs = address_info['countTxs']

            lifetime = get_lifetime(trans_timestamps) 
            addr_get_profit = get_addr_get_profit(transactions_data) 
            avg_fee = get_avg_fee(transactions_data) 
            max_send = get_max_send(transactions_data)
            giniV = np.array(get_gini_value(transactions_data))
            gini_value = gini_co(giniV) 
            
            features.append({
                'address': address,
                "num_instructions": num_instructions,
                "num_unique_instructions": num_unique_instructions,
                "num_jumps": num_jumps,
                "num_cond_jumps": num_cond_jumps,
                "num_function_calls": num_function_calls,
                "num_external_calls": num_external_calls,
                "num_internal_calls": num_internal_calls,
                "num_push_instructions": num_push_instructions,
                "num_pop_instructions": num_pop_instructions,
                "num_stack_operations": num_stack_operations,
                "num_sload_sstore": num_sload_sstore,
                "num_arith_logic": num_arith_logic,
                "stack_depth": max_stack_depth,
                "code_size": code_size,
                "calling_instructions_ratio": calling_instructions_ratio, 
                "timestamp_dependency": timestamp_dependency, 
                "cyclomatic_complexity": cyclomatic_complexity,
                'balance': balance,
                'totalIn': totalIn,
                'totalOut': totalOut,
                'MaxSend': max_send,
                'contTxs': countTxs,            
                'AvgFee': avg_fee,
                "gas_cost": gas_cost,
                "avg_gas_cost_per_instr": avg_gas_cost_per_instr,            
                'lifetime': lifetime,
                'addr_get_profit':addr_get_profit,
                'gini_value': gini_value,
                "bytecode": bytecode_hex,
                "opcodes": opcodes,
                "instruction_counts": instruction_counts,
            })

        except Exception as e:
            print(f"Error processing address {address}: {str(e)}")
            continue

    return features

#####################################
if __name__ == '__main__':
    try:
        addresses = read_addresses_from_csv('LegitContracts.csv') #Read The CSV File that contain only Addresses
        api_key = random.choice([API_KEY1, API_KEY2])
        data = extract_data(addresses, api_key)
        write_data_to_csv(data, 'LegitFeaturesComplete.csv') #Export the Features
        print(f"Completed at: {datetime.now()}")
    except Exception as e:
        print(f"Error during main execution: {str(e)}")    
#####################################

