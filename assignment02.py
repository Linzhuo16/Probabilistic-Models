import random



random.seed(12345)
english_words = []
with open("Lab02TextFiles/words_alpha.txt") as word_file:
    for word in word_file:
        length = len(word.strip().lower())
        if(length == 4):
            english_words.append(word.strip().lower())


def is_english_word(word):
    return word.lower() in english_words

def problem_a():
    words = []
    for i in range(100):
        word = []
        for j in range(4):
            str = random.randint(1, 26)
            word.append(chr(str + 96))
        words.append(word)

    count = 0
    for word in words:
        w = ''
        for char in word:
            w += char
            print(char,end='')
        if(is_english_word(w)):
            count += 1
        print("")
    print(count)
    return count

def problem_b(file):
    chars = [0 for i in range(26)]

    f = open(file)
    for line in f:
        for str in line:
            i = ord(str)
            if( 97<=i<=122):
                chars[i - 97] += 1
            elif( 65 <= i<=90):
                chars[i-65] +=1
    f.close()

    prob = [0 for i in range(26)]
    prob[0] = chars[0]
    for i in range(26):
        prob[i] = prob[i-1] + chars[i]
    len = prob[-1]

    words = []
    for i in range(100):
        word = []
        for j in range(4):
            x = random.randint(1, len)
            for k in range(26):
                if (x <= prob[k]):
                    word.append(chr(k+97))
                    break;

        words.append(word)

    count = 0
    for word in words:
        w = ''
        for char in word:
            w += char
            print(char, end='')
        if (is_english_word(w)):
            count += 1
        print("")
    print(count)
    return count


def random_str(file):
    chars = [0 for i in range(26)]
    f = open(file)
    for line in f:
        for str in line:
            i = ord(str)
            if (97 <= i <= 122):
                chars[i - 97] += 1
            elif (65 <= i <= 90):
                chars[i - 65] += 1
    f.close()
    prob = [0 for i in range(26)]
    prob[0] = chars[0]
    for i in range(26):
        prob[i] = prob[i - 1] + chars[i]
    return prob

def problem_c(file):
    chars = [[0 for i in range(26)] for i in range(26)]
    #print(chars)

    f = open(file)
    for line in f:
        line.replace("-", "")
        words = line.split(' ')
        for word in words:
            length = len(word)
            if(length<2): break
            for i in range(1, length):
                x_1 = ord(word[i])
                x_2 = ord(word[i-1])

                if( 97<=x_1<=122):
                    x_1 -= 97
                elif( 65 <= x_1 <=90):
                    x_1 -= 65
                else:
                    break;

                if (97 <= x_2 <= 122):
                    x_2 -= 97
                elif (65 <= x_2  <= 90):
                    x_2 -= 65
                else:
                    break;

                chars[x_2][x_1] += 1
    f.close()
    #print(chars)

    prob = [[0 for i in range(26)] for i in range(26)]
    length = [0 for i in range(26)]

    for j in range(26):
        prob[j][0] = chars[j][0]
        for i in range(26):
            prob[j][i] = prob[j][i - 1] + chars[j][i]
        length[j] = prob[j][-1]


    prob_b  = random_str(file)
    len_b = prob_b[-1]

    words = []
    for i in range(100):
        word = []
        x = random.randint(1, len_b)
        #print("")
        for k in range(26):
            if (x <= prob_b[k]):
                word.append(chr(k + 97))
                break
        #print("1")
        for j in range(3):
           # print(word[j])
            x_2 = ord(word[j]) - 97
            #default
            if(length[x_2] == 0):
                x = random.randint(1, len_b)
                for k in range(26):
                    if (x <= prob_b[k]):
                        word.append(chr(k + 97))
                        break

            else:
                x = random.randint(1, length[x_2])
                for k in range(26):
                    if (x <= prob[x_2][k]):
                        word.append(chr(k + 97))
                        break;

        words.append(word)

    count = 0
    for word in words:
        w = ''
        for char in word:
            w += char
            print(char, end='')
        if (is_english_word(w)):
            count += 1
        print("")
    print(count)
    return count

def random_str_two(file):
    chars = [[0 for i in range(26)] for i in range(26)]
    # print(chars)

    f = open(file)
    for line in f:
        line.replace("-", "")
        words = line.split(' ')
        for word in words:
            length = len(word)
            if (length < 2): break
            for i in range(1, length):
                x_1 = ord(word[i])
                x_2 = ord(word[i - 1])

                if (97 <= x_1 <= 122):
                    x_1 -= 97
                elif (65 <= x_1 <= 90):
                    x_1 -= 65
                else:
                    break;

                if (97 <= x_2 <= 122):
                    x_2 -= 97
                elif (65 <= x_2 <= 90):
                    x_2 -= 65
                else:
                    break;

                chars[x_2][x_1] += 1
    f.close()
    # print(chars)

    prob = [[0 for i in range(26)] for i in range(26)]
    length = [0 for i in range(26)]

    for j in range(26):
        prob[j][0] = chars[j][0]
        for i in range(26):
            prob[j][i] = prob[j][i - 1] + chars[j][i]
        length[j] = prob[j][-1]

    return  prob, length


def problem_d(file):
    chars = [[[0 for i in range(26)] for i in range(26)]for i in range(26)]
    f = open(file)
    for line in f:
        line.replace("-", "")
        words = line.split(' ')
        for word in words:
            length = len(word)
            if (length < 3): break
            for i in range(2, length):
                x_1 = ord(word[i])
                x_2 = ord(word[i - 1])
                x_3 = ord(word[i - 2])

                if (97 <= x_1 <= 122):
                    x_1 -= 97
                elif (65 <= x_1 <= 90):
                    x_1 -= 65
                else:
                    break;

                if (97 <= x_2 <= 122):
                    x_2 -= 97
                elif (65 <= x_2 <= 90):
                    x_2 -= 65
                else:
                    break;

                if (97 <= x_3 <= 122):
                    x_3 -= 97
                elif (65 <= x_3 <= 90):
                    x_3 -= 65
                else:
                    break;

                chars[x_3][x_2][x_1] += 1
    f.close()

    prob = [[[0 for i in range(26)] for i in range(26)] for i in range(26)]
    length = [[0 for i in range(26)] for i in range(26)]

    for k in range(26):
        for j in range(26):
            prob[k][j][0] = chars[k][j][0]
            for i in range(26):
                prob[k][j][i] = prob[k][j][i - 1] + chars[k][j][i]

            length[k][j] = prob[k][j][-1]

    prob_b = random_str(file)
    len_b = prob_b[-1]

    prob_c, length_c = random_str_two(file)


    words = []
    for i in range(100):
        word = []
        x = random.randint(1, len_b)
        for k in range(26):
            if (x <= prob_b[k]):
                word.append(chr(k + 97))
                break


        x_2 =  ord(word[0]) - 97
        # default
        if (length_c[x_2] == 0):
            x = random.randint(1, len_b)
            for k in range(26):
                if (x <= prob_b[k]):
                    word.append(chr(k + 97))
                    break

        else:
            x = random.randint(1, length_c[x_2])
            for k in range(26):
                if (x <= prob_c[x_2][k]):
                    word.append(chr(k + 97))
                    break;


        for j in range(2):

            x_3 = ord(word[j]) - 97
            x_2 = ord(word[j+1]) - 97
            # default
            if (length[x_3][x_2] == 0):
                x = random.randint(1, len_b)
                for k in range(26):
                    if (x <= prob_b[k]):
                        word.append(chr(k + 97))
                        break

            else:
                x = random.randint(1, length[x_3][x_2])
                for k in range(26):
                    if (x <= prob[x_3][x_2][k]):
                        word.append(chr(k + 97))
                        break;

        words.append(word)

    count = 0
    for word in words:
        w = ''
        for char in word:
            w += char
            print(char, end='')
        if (is_english_word(w)):
            count += 1
        print("")
    print(count)
    return count

if(__name__ == "__main__"):
    filepath1 = "Lab02TextFiles/spamiam.txt"
    filepath2 = "Lab02TextFiles/saki_story.txt"

    a = problem_a()
    b1 = problem_b(filepath1)
    b2 = problem_b(filepath2)
    c1 = problem_c(filepath1)
    c2 = problem_c(filepath2)
    d1 = problem_d(filepath1)
    d2 = problem_d(filepath2)
    # print(a, b1, b2, c1, c2, d1, d2)