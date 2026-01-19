import sys

def next_char(parse):
    i = 0
    while i < len(parse) and parse[i].isspace():
        i += 1
    if i >= len(parse):
        raise Exception("unexpected end", parse)
    return parse[i], parse[i+1:]


def read_label(parse):
    ch, rest = next_char(parse)
    if ch in "()":
        raise Exception("invalid label", parse)

    i = 0
    while i < len(rest) and not rest[i].isspace() and rest[i] not in "()":
        i += 1

    label = ch + rest[:i]
    if not label:
        raise Exception("empty label", parse)

    return rest[i:], label


def read_tree(parse, word_list=None, const_list=None):
    if word_list is None:
        word_list = []
    if const_list is None:
        const_list = []

    ch, parse = next_char(parse)
    if ch != "(":
        raise Exception("expected '('", parse)

    parse, label = read_label(parse)

    start = len(word_list)
    constituent = [label, start, None]
    const_list.append(constituent)

    if next_char(parse)[0] == "(":
        while next_char(parse)[0] != ")":
            parse, word_list, const_list = read_tree(parse, word_list, const_list)
        _, parse = next_char(parse)
    else:
        parse, word = read_label(parse)
        word_list.append(word)
        ch, parse = next_char(parse)
        if ch != ")":
            raise Exception("expected ')' after word", parse)

    constituent[2] = len(word_list)
    return parse, word_list, const_list


def build_tree(parse, word_list, const_list):
    label, start, end = const_list.pop(0)
    sublabels = label.split("=")
    parse += "(" + "(".join(sublabels)

    if end == start + 1:
        parse += " " + word_list[start] + ")" * len(sublabels)
        return parse, word_list, const_list

    while const_list and const_list[0][1] < end:
        parse, word_list, const_list = build_tree(parse, word_list, const_list)

    parse += ")" * len(sublabels)
    return parse, word_list, const_list


def parse(parse_string):
    try:
        rest, word_list, const_list = read_tree(parse_string)
        if rest.strip():
            raise Exception("trailing characters", rest)

        i = 0
        collapsed = []
        while i < len(const_list):
            label, s, e = const_list[i]
            j = i + 1
            while j < len(const_list) and const_list[j][1] == s and const_list[j][2] == e:
                label = label + "=" + const_list[j][0]
                j += 1
            collapsed.append((label, s, e))
            i = j
        const_list = collapsed

        return parse_string, word_list, const_list

    except Exception as e:
        message, rest = e.args
        pos = len(parse_string) - len(rest)
        print(parse_string)
        print(" " * pos + "^")
        print(message)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tree-reader.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            print("ORIGINAL")
            print(line)

            result = parse(line)
            if result is None:
                print()
                continue

            original, words, consts = result
            print("WORDS")
            print(words)

            print("CONSTITUENTS")
            print(consts)
            
            # Rebuild tree for display
            tree, _, _ = build_tree("", words, consts.copy())
            print("REBUILT")
            print(tree)
            print()
