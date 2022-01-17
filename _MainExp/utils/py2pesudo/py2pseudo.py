'''
INSTRUCTIONS
1. Create a file with the following code
2. Put the file you want to convert into the same folder as it, and rename it to "file.py"
3. Add a "#F" comment to any lines in the code which have a function call that doesn't assign anything (so no =),
as the program cannot handle these convincingly
4. Run the converter file
'''

import re

python_file = './target.py'
work_file = None

basic_conversion_rules = {"for": "FOR", "=": "TO", "if": "IF", "==": "EQUALS", "while": "WHILE", "until": "UNTIL", "import": "IMPORT", "class": "DEFINE CLASS", "def": "DEFINE FUNCTION", "else:": "ELSE:", "elif": "ELSEIF", "except:": "EXCEPT:", "try:": "TRY:", "pass": "PASS", "in": "IN"}
prefix_conversion_rules = {"=": "SET ", "#F": "CALL "}
advanced_conversion_rules = {"print": "OUTPUT", "return": "RETURN", "input": "INPUT"}

def f2list(to_list):
    return to_list.readlines()

def l2pseudo(to_pseudo):
    for line in to_pseudo:
        line_index = to_pseudo.index(line)
        line = str(line)
        line = re.split(r'(\s+)', line)
        for key, value in prefix_conversion_rules.items():
            if key in line:
                if not str(line[0]) == '':
                    line[0] = value + line[0]
                else:
                    line[2] = value + line[2]
        for key, value in basic_conversion_rules.items():
            for word in line:
                if key == str(word):
                    line[line.index(word)] = value
        for key, value in advanced_conversion_rules.items():
            for word in line:
                line[line.index(word)] = word.replace(key, value)
        for key, value in prefix_conversion_rules.items():
            for word in line:
                if word == key:
                    del line[line.index(word)]
        to_pseudo[line_index]= "".join(line)
    return(to_pseudo)

def p2file(to_file):
    file = open(python_file + '_pseudo.txt', 'w', encoding="utf8")
    for line in to_file:
        print(line, file=file)

def main():
    main_file = open(python_file, 'r+')
    work_file = f2list(main_file)
    work_file = l2pseudo(work_file)
    p2file(work_file)
    

main()