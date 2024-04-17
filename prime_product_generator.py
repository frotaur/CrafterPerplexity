"""
    Script that generates the .txt dataset of prime products. 
"""


import math, random
from tqdm import tqdm
from modules.tok_utils import tok_text_to_h5
# ------------------- PARAMETERS, CHANGE HERE -------------------
n_pairs = 10**8 # Number of examples to generate
(min_n_digits, max_n_digits) = (1, 5) # Digit range of primes, inclusive
mul_space, eq_space = (' ', ' ' * 3) # space before/after '*', space before/after '=' 
input_file_name = 'list-of-prime-numbers.txt' # List of prime numbers to draw with. Goes up to 19 999 999.
# ------------------- DO NOT CHANGE BELOW -------------------
output_file_name = f'prime-products_{min_n_digits}_{max_n_digits}.txt'

def convert_to_int(s):
    try: return int(s.strip())
    except: return 0

def load_primes(file_name):
    file = open(file_name)
    lines = file.readlines()
    primes = [convert_to_int(line.strip()) for line in lines]
    return primes

def load_long_primes(file_name):
    long_primes = [p for p in load_primes(file_name) if p > 0 and min_n_digits <= math.ceil(math.log10(p)) <= max_n_digits]
    return long_primes

def generate_pair(primes):
    [first_prime, second_prime] = [random.choice(primes) for i in range(2)]
    while first_prime == second_prime:
        [first_prime, second_prime] = [random.choice(primes) for i in range(2)]
    return (first_prime, second_prime) if first_prime < second_prime else (second_prime, first_prime)


def generate_prime_mult_line(primes, add_new_line=False):
    (first_prime, second_prime) = generate_pair(primes)
    product = first_prime * second_prime
    n_digits = 2 * max_n_digits
    prime_mult_line = f"{first_prime:0{max_n_digits}}{mul_space}*{mul_space}{second_prime:0{max_n_digits}}{eq_space}={eq_space}"+f"{product:0{n_digits}d}"[::-1]
    # print('line size :' ,len(prime_mult_line))
    # print('line : ', prime_mult_line)
    if add_new_line: prime_mult_line += '\n'
    return prime_mult_line


def generate_file(input_file_name, output_file_name, n_lines):
    long_primes = load_long_primes(input_file_name)
    lines = [generate_prime_mult_line(long_primes, True) for i in tqdm(range(n_lines))]
    with open(output_file_name, 'w') as file: file.writelines(lines)

def generate_file_low_memory(input_file_name, output_file_name, n_lines):
    long_primes = load_long_primes(input_file_name)
    with open(output_file_name, 'w') as file:
        for _ in tqdm(range(n_lines)):
            file.write(generate_prime_mult_line(long_primes, True))

def run_main():
    generate_file_low_memory(input_file_name, output_file_name, n_pairs)

if __name__ == '__main__': 
    run_main()
    tok_text_to_h5(output_file_name,None,tokenizer_folder='tokenizers',tokenizer_name='primes',delete_txt=True,output_name='primes_1_5')