#!/usr/bin/env python3
"""Fake SSH/register helper for subprocess tests; never opens a network socket."""

import json
import os
import subprocess
import sys
import time


def main():
    if "--host" in sys.argv:
        if sys.argv[-2:] != ["--register", "420"]:
            raise SystemExit(17)
        command, output = "register 420", '{"register420":63}'
    else:
        for destination, password in json.loads(os.getenv("STEM_CAMERA_SSH_PASSWORDS", "{}")).items():
            answer = subprocess.run([os.environ["SSH_ASKPASS"], destination + "'s password: "],
                                    capture_output=True, text=True, check=True)
            if answer.stdout != password + "\n":
                raise SystemExit(17)
        command = sys.stdin.read().strip()
        if command == "dsh sensor temp":
            output = "Q1 : 44.63\nQ2 : 41.91\nQ3 : 54.62\nQ4 : 46.05\nB1 : 36.06\nB2 : 35.53\nHS1 : 32.46\nHS2 : 32.76"
        elif command == "dsh bias adc volts":
            output = "AVDD : 2.500 V\nVDDR : 1.802 V\nBIASN : 0.400 mA\nN3BUF : 4.020 mA"
        else:
            raise SystemExit(17)
    with open(os.environ["STEM_TEST_HARDWARE_AUDIT"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
    time.sleep(.15)
    print(output)


if __name__ == "__main__":
    main()
