#!/usr/bin/env python3

import sys
import os
import argparse
import jitu
from collections import defaultdict
import subprocess
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
fold2dotbracket_path = os.path.join(SCRIPT_DIR, "fold2dotbracketFasta.py")
rnaConvert_path = os.path.join(SCRIPT_DIR, "rnaConvert.py")

def generate_constraints(ref, profile):
    state = []
    for pos, (base, bit) in enumerate(zip(ref, profile), 1):
        if bit == '1':
            state.append([str(pos), base, '0'])
        else:
            state.append([str(pos), base, '-1'])
    return state

def handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bit_file", type=str)
    parser.add_argument("-r", "--reference_file", type=str)
    parser.add_argument("-t", "--transcript", type=str)
    parser.add_argument("-s", "--size_file", type=str)

    parser.set_defaults(
        bit_file='merged_R1_R2.bit',
        reference_file='cool6.fasta',
        transcript='COOLAIR3',
        size_file='sizer.tab'
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = handler()

    tube, seqD = jitu.getTubeD(fastaFiler=args.reference_file)
    reference_sequence = seqD[args.transcript]

    unique_bits = {}
    profileD = defaultdict(int)
    tot = 0

    with open(args.bit_file) as inp:
        for line in inp:
            A = line.strip().split('\t')
            if A:
                bits = ''.join(['1' if b == '1' else '.' for b in A[1:]])
                unique_bits[bits] = bits.count('1')
                profileD[bits] += 1
                tot += 1

    print(f"unique bits vectors: {len(unique_bits)} from total: {tot}")

    merged_db = open("merged.db", "w")
    merged_txt = open("merged.txt", "w")
    merged_el = open("merged.element_string", "w")
    outS = open(args.size_file, "w")

    seen = defaultdict(int)

    for t, (profile, num_mutations) in enumerate(
        tqdm(sorted(unique_bits.items(), key=lambda x: x[1], reverse=True),
            desc="Time remaining until you can get taco bell"),
        1
    ):
        assert len(reference_sequence) == len(profile), "Reference length mismatch!"

        state = generate_constraints(reference_sequence, profile)
        ticks_string = ''.join([x[-1] for x in state])
        seen[ticks_string] += 1

        if seen[ticks_string] > 1:
            continue  # skip identical constraints

        bit_prefix = f"bit_{t}"

        constraint_temp = bit_prefix + ".tmp.bpseq"
        with open(constraint_temp, "w") as outf:
            for i, base, tick in state:
                outf.write(f"{i}\t{base}\t{tick}\n")

        fold_out = bit_prefix + ".fold"
        post_out = bit_prefix + ".post"

        subprocess.run(
            ["contrafold", "predict", "--constraints", constraint_temp,
             "--parens", fold_out, "--posteriors", "0.0", post_out],
            capture_output=True, text=True
        )

        db_out = bit_prefix + ".db"
        subprocess.run(
            ["python", fold2dotbracket_path, "--input_file", fold_out,
             "--tag", bit_prefix, "--output_file", db_out],
            capture_output=True, text=True
        )

        with open(db_out) as f:
            dotbracket = f.read().strip()

        merged_db.write(dotbracket + "\n")

        subprocess.run(
            ["python", rnaConvert_path, db_out, "-T", "element_string",
             "--force", "--to-file", "--filename", bit_prefix + ".txt"],
            capture_output=True, text=True
        )

        element_file = bit_prefix + ".txt001.element_string"
        with open(element_file) as inp:
            for line_num, line in enumerate(inp, 1):
                if line_num == 3:
                    merged_el.write(f"{bit_prefix}\t{line.strip()}\n")
                    break

        with open(post_out) as h:
            txt_content = h.read().strip()
            merged_txt.write(f"{bit_prefix}\t{txt_content}\n")

        outS.write(f"{bit_prefix}\t{profileD[profile]}\t{profile}\n")

        os.remove(constraint_temp)
        os.remove(fold_out)
        os.remove(post_out)
        os.remove(db_out)
        os.remove(element_file)

        tmp_txt = bit_prefix + ".txt"
        if os.path.exists(tmp_txt):
            os.remove(tmp_text)

    merged_db.close()
    merged_txt.close()
    merged_el.close()
    outS.close()

    print("DONE")

