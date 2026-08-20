import argparse
import colorama
import json
import os
from pathlib import Path


def delete_from_q(args):
    q_list = os.listdir(args.q_path)
    for i in q_list:
        print(colorama.Fore.YELLOW + i + colorama.Fore.RESET)
        while True:
            inp = input("Delete[d] or Keep[k]? ").lower().strip()
            if inp and inp in ["d", "k"]:
                break
        
        q_file_path = args.q_path / i
        s_file_path = args.s_path / i

        if inp == "d":
            if q_file_path.exists():
                os.remove(q_file_path)
            if s_file_path.exists():
                os.remove(s_file_path)
            print(colorama.Fore.RED + i + " is Deleted." + colorama.Fore.RESET)
        elif inp == "k":
            if q_file_path.exists():
                os.remove(q_file_path)
            print(colorama.Fore.GREEN + i + " is Restored." + colorama.Fore.RESET)

def delete_from_json(args):
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in data:
        print(colorama.Fore.YELLOW + i["path"] + " | " + "Score: " + str(i["score"]) + colorama.Fore.RESET)
        print()

        for j in i.get("sample_lines", []):
            print(colorama.Fore.RED + j + colorama.Fore.RESET)
        
        print(f"First {str(args.lines)} lines:")
        with open(args.s_path / i["path"], "r", encoding="utf-8", errors="ignore") as f:
            for k in f.read().splitlines()[:args.lines]:
               print(k)


        print()
        while True:
            inp = input("Delete[d] or Keep[k]? ").lower().strip()
            if inp and inp in ["d", "k"]:
                break
                
        # Resolve paths dynamically 
        s_file_path = args.s_path / i["path"]

        if inp == "d":
            if s_file_path.exists():
                os.remove(s_file_path)
            print(colorama.Fore.RED + i["path"] + " is Deleted." + colorama.Fore.RESET)
        elif inp == "k":
            print(colorama.Fore.GREEN + i["path"] + " is Restored." + colorama.Fore.RESET)

def main():
    colorama.init()
    print("="*40)
    print()
    print("Started Fixing Quarantines")
    print()
    print("="*40)

    parser = argparse.ArgumentParser(description="Quarantine Fixer")
    parser.add_argument("--q-path", type=Path, default=None, help="Quarantine Path")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--s-path", type=Path, required=True,  help="Source Path")
    parser.add_argument("--lines", type=int, default=10, help="Number of lines to display from the file")

    args = parser.parse_args()
    if args.q_path:
        delete_from_q(args)
    else:
        delete_from_json(args)

if __name__ == "__main__":
    main()

