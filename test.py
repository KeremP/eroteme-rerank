#!/usr/bin/env python
from handler import lambda_handler
from pprint import PrettyPrinter

if __name__ == "__main__":
    EVENT = {
        "urls":["https://en.wikipedia.org/wiki/Elon_Musk"],
        "query":"Where was Elon born?"
    }

    pp = PrettyPrinter(indent=4)

    resp = lambda_handler(EVENT, None)

    pp.pprint(resp)