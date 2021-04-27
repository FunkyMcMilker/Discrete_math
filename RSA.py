import numpy as np
from numpy.linalg import det, inv
import sympy.ntheory as nt
print("# Q01")
print("# Suppose you are running a small online shopping business.")
print("# You are planning to use a double security system -- ")
print("# ``RSA - Matrix Multiplication Modulo P'' where P is a")
print("# prime number -- for message security for your clients.")
print("# Write a small Python program to demonstrate how a client's plaintext")
print("# ``MATH 2410 at Webster University Bangkok, Monday 12:30pm''")
print("# would be two-step-encrypted into ciphertext, sent over the internet")
print("# and, once arrived, two-step-decrypted back to the plaintext by you.")
# Use the following functions in your programs.


# Return list of postions in l where n are.
def postn(n, l):  # This definition uses list comprehension.
    return [x for x in range(len(l)) if l[x] == n]


# Extended Euclidean algorithm
# Bezout Thm
def egcd(a, b):
    if a == 0:
        return 0, 1, b
    else:
        s, t, gcd = egcd(b % a, a)
        return t - (b // a) * s, s, gcd


# s, t, gcd = egcd(30, 50)
# print(s, t, gcd)
# From Textbook, Ch 04
# Algorithm 5 Fast Modular Exponentiation Sec 4.2
def modExp(b, n, m):
    a = "{0:0b}".format(n)
    # print(a)
    x = 1
    power = b % m
    for i in range(len(a) - 1, 0 - 1, -1):
        if a[i] == '1':
            x = (x * power) % m
            # print(i,":",x)
        power = (power * power) % m
    return x  # x = b^n mod m


# MinSet removes repetition of elements in a set.
def MinSet(set):
    mnst = []
    for i in range(len(set)):
        if not (set[i] in mnst):
            mnst.append(set[i])
    return mnst


# Prime factorization
def factor(n):
    d = 2
    factors = []
    while n >= d * d:
        if n % d == 0:
            n = int(n / d)
            factors.append(d)
        else:
            d = d + 1
    if n > 1:
        factors.append(n)
    return factors


# Is n a prime number?
def isPrime(n):
    return len(factor(n)) == 1


# List all divisors of n.
def divisors(n):
    return [x for x in list(range(1, n + 1)) if n % x == 0]


# Return quotient and remainder of m/n.
def DivMod(m, n):
    return m // n, m % n


# Euler's totient function
def phi(n):
    fctr_lst = factor(n)
    distinct_primes = MinSet(factor(n))
    ph = 1
    for p in distinct_primes:
        e = len([x for x in fctr_lst if x == p])
        ph *= (p ** e - p ** (e - 1))
    return ph


# Return an n-digit format of integer i.
def base10(i, n):
    return ("{0:0" + str(n) + "d}").format(i)


# base10(0,2)
def matrix2list(M):
    list = []
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            list.append(int(M[i][j]))
    return list


def list2matrix(l, s):
    M = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            M[i][j] = l[i * s[1] + j]
    return M


def list2text(l):
    return ''.join([str(alphabetSet[x]) for x in l])


def text2list(s):
    return [postn(s[i], alphabetSet)[0] for i in range(len(s))]


def text2numstring(s):
    return ''.join([base10(postn(s[i], alphabetSet)[0], 2) for i in range(len(s))])


def num2textstring(s):
    return ''.join([alphabetSet[10 * int(s[2 * i]) + int(s[2 * i + 1])] for i in range(len(s) // 2)])


# Multiply matrices modulo 73
def mul(X, Y):
    Z = np.zeros((len(X), len(Y[0])))
    # iterate through rows of X
    for i in range(len(X)):
        # iterate through columns of Y
        for j in range(len(Y[0])):
            # iterate through rows of Y
            for k in range(len(Y)):
                Z[i][j] += int(round(X[i][k] * Y[k][j], 0))
    return Z % 73


# Inverse matrix of M modulo m
def invMatmodm(M, m):
    s, t, gcd = egcd(m, int(round(det(M), 0)))
    print("M:m =")
    print(M, ":", m)
    print("s =", s)
    print("t =", t)
    print("gcd =", gcd)
    return t * det(M) * inv(M) % m


print("#")
print("#")
print("#")
print("# Encryption Level 01 starts here:")
print("#")
print("#")
print("#")
alphabetSet = " ABCDEFGHIJKLMNOPQRSTUVWXYZ,.?:'/-_abcdefghijklmnopqrstuvwxyz0123456789+="
plain_text_level_00 = "MATH 2410 at Webster University Bangkok, Monday 12:30 pm // "
print("\nalphabetSet is: '" + str(alphabetSet) + "'.")
print("\nPlain Text is: ", plain_text_level_00)
plain_text_num = text2numstring(plain_text_level_00)
print("\nPlain Text in number form is: ", plain_text_num)
plain_text_num_list = text2list(plain_text_level_00)
print("\nPlain Text in list form is: ", plain_text_num_list)
M00 = list2matrix(plain_text_num_list, (19, 3))
print("\nPlain Text in n-by-3 matrix form is: ")
print(M00)
encodeMatrix = np.array([[1, 2, -1], [2, 0, 1], [1, -1, 0]])  # np.identity(3)
print("\na 3-by-3 encoding matrix used is: ")
print(encodeMatrix)
cipher_level_01_matrix = mul(M00, encodeMatrix)
print("\ncipher_level_01_matrix is: ")
print(cipher_level_01_matrix)
# print(mul(cipher_level_01_matrix,invMatmodm(encodeMatrix,73)))
cipher_level_01_list = matrix2list(cipher_level_01_matrix)
print("\ncipher_level_01_list is: ", cipher_level_01_list)
cipher_level_01_text = list2text(cipher_level_01_list)
print("\ncipher_level_01_text is: ", cipher_level_01_text)
cipher_level_01_number = text2numstring(cipher_level_01_text)
print("\ncipher_level_01_number is: ", cipher_level_01_number)
print("#")
print("#")
print("#")
print("# Encryption Level 02 starts here:")
print("#")
print("#")
print("#")
# ''.join(['1234'[i] for i in range(0,2)])
CM01 = ''.join([cipher_level_01_number[i] for i in range(0, 38)])
CM02 = ''.join([cipher_level_01_number[i] for i in range(38, 76)])
CM03 = ''.join([cipher_level_01_number[i] for i in range(76, 114)])
print("\nThe level one encrypted message in number form ")
print("is split into 3 pieces: ")
print("first part:", CM01)
print("second part:", CM02)
print("third part:", CM03)
"""
To see how difficult to decrypt the ciphertext without d, we try another
numerical example using bigger prime numbers.
Example: choose p = 59 649 589 127 497 217 and q = 5 704 689 200 685 129␣
,→054 721
(2 distinct primes).
n = p*q = 340282366920938463463374607431768211457 and
phi(n) = 340282366920938457758625757157511659520.
For simplicity, choose e such that gcd(e,phi(n)) = 1, we choose e =␣
,→10007,
and we found
d = 100483101254259332734759579534370635943.
The company publish {e,n} =␣
,→{10007,340282366920938463463374607431768211457}.
"""
print("\nI have chosen p as a 17-digit prime and q as a 22-digit prime.")
print("(You have to choose your own more-than-22-digit primes. Hint: Use available online resources.)")
p = 59_649_589_127_497_217
q = 5_704_689_200_685_129_054_721
print("\n'p is prime' is", nt.isprime(p), ".")
print("\n'q is prime' is", nt.isprime(q), ".")
n__ = p * q  # 340282366920938463463374607431768211457
print("\nn is pq = ", n__, ".")
# sympy.totient(n__) # This phi(n) cannot be easily computed from n:
# phi(n__) # or my version of phi(n).
ph__ = (p - 1) * (q - 1)  # 340282366920938457758625757157511659520
print("\nphi(n) is not feasibly obtained from n,")
print("but it can be easily obtained from p and q as")
print("phi(n)=(p-1)*(q-1) = ", ph__, "(from property of n=p*q).")
print("\nFor simplicity, choose e such that gcd(e,phi(n)) = 1,")
print(" we choose e = 10007, and we found")
print("d = 100483101254259332734759579534370635943 from the Bezout Thm,")
print("or extended Euclid Algorithm.")
e__ = 10007  # public key: {e,n}
s__, t__, gcd__ = egcd(ph__, e__)
print("\n(" + str(t__) + ")(" + str(e__) + ")+(" + str(s__) + ")(" + str(ph__) + ") = "
      + str(gcd__) + ".")
d__ = 100483101254259332734759579534370635943  # private key: {d,n}
print("\nIf you mod both sides of the above equation with phi(n),")
print("where phi(n) = 340282366920938457758625757157511659520,")
print("you will see that d*e = 1 (mod phi(n)) or d*e = k*phi(n) + 1,")
print("where k is an integer.")
print("This is the required property for C^d = (M^e)^d = M^(phi(n)*k)*M^1 mod(n),")
print("since the totient function phi(n) has a property that M^phi(n) = 1 (mod n),")
print("then C^d = 1^k*M = M mod(n), and you will recover the message sent over")
print("the internet, M, but this is our level-1-ciphered message.")
print("\nThe company, you, will post only the public key {e, n} and")
print("keep the private key d privately for your own decryption.")
print("\nThe level one ciphertext were split into three chunks:")
M__ = int(CM01)  # ciphertext level 01
print("M01 =", M__, ", and encrypted by the client by the public key,")
C__1 = modExp(M__, e__, n__)  # ciphertext level 02 (encrypted text using public key)
print("and sent over the internet as CM01_L2 =", C__1, ".")
DM__1 = str(modExp(C__1, d__, n__))  # decryption result
C__1 = str(C__1)
# C__1 = '0'*(38-len(C__1))+C__1
DM__1 = str(DM__1)
DM__1 = '0' * (38 - len(DM__1)) + DM__1
print("Once received by you, it then is decrypted by the private key to be DM01 =", DM__1, ".")
M__ = int(CM02)  # ciphertext level 01
print("M02 =", M__, ", and encrypted by the client by the public key,")
C__2 = modExp(M__, e__, n__)  # ciphertext level 02 (encrypted text using public key)
print("and sent over the internet as CM02_L2 =", C__2, ".")
DM__2 = modExp(C__2, d__, n__)  # decryption result
C__2 = str(C__2)
# C__2 = '0'*(38-len(C__2))+C__2
DM__2 = str(DM__2)
DM__2 = '0' * (38 - len(DM__2)) + DM__2
print("Once received by you, it then is decrypted by the private key to be DM02 =", DM__2, ".")
M__ = int(CM03)  # ciphertext level 01
print("M03 =", M__, ", and encrypted by the client by the public key,")
C__3 = modExp(M__, e__, n__)  # ciphertext level 02 (encrypted text using public key)
print("and sent over the internet as CM03_L2 =", C__3, ".")
DM__3 = modExp(C__3, d__, n__)  # decryption result
C__3 = str(C__3)
# C__3 = '0'*(38-len(C__3))+C__3
DM__3 = str(DM__3)
DM__3 = '0' * (38 - len(DM__3)) + DM__3
print("Once received by you, it then is decrypted by the private key to be DM03 =", DM__3, ".")
print("\nThen the three decrypted messages are spliced in postion number format as")
# Cipher_L2 = C__1+C__2+C__3
DM = DM__1 + DM__2 + DM__3
# print(Cipher_L2)
print(DM, ".")
print("In text form it is ", num2textstring(DM), "which is still unreadable, since")
print("it is a level one ciphertext, where its text in list format is")
print(text2list(num2textstring(DM)), ".")
DM_matrix_02 = list2matrix(text2list(num2textstring(DM)), (19, 3))
print("then the list is put in n-by-3 matrix as")
print(DM_matrix_02, ",")
DM_matrix_01 = mul(DM_matrix_02, invMatmodm(encodeMatrix, 73))
print("and decrypted another level by the inverse matrix of the encrypt matrix modulo P=73")
print("to be the matrix of plaintext in postion number format:")
print(DM_matrix_01, ",")
print("next the matrix is converted back to a list of position numbers: ", matrix2list(DM_matrix_01))
print("in which the plaintext sent by the client: ", list2text(matrix2list(DM_matrix_01)))
print("can be read by only you through the list of alphabet characters agreed to be used by")
print("both your shop and your clients: ")
print("'" + str(alphabetSet) + "'.")
