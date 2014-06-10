import argparse, sys
import itertools
import random

def main():
	parser = argparse.ArgumentParser(description='Periodic sequence generator.')
	parser.add_argument('-l', nargs='?', help='sequence length (default: 10000)', type=int, default=10000)
	parser.add_argument('-p', help='period length', type=int)
	parser.add_argument('-o', help='output file', type=argparse.FileType('w', 0))
	args = parser.parse_args(sys.argv[1:])

	period = []
	for _ in itertools.repeat(None, args.p):
		period.append(chr(65+random.randint(0, 4)))
	for _ in itertools.repeat(None, args.l/args.p):
		for i in xrange(args.p):
			args.o.write("%s\n"%period[i])

if __name__ == "__main__":
	main()