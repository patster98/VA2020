import random
randomN = random.randrange(0,11,2)
# print(randomN)

#main
while True:
    guess = int(input("Please enter a number from 0 to 10: "))
    if guess == randomN:
        print("Congratulations! Your number is:", "" , guess)
        break
    else:
        print("Wrong number")
        check = input ("Continue? Y or N: ")
        if check == "Y" or check == "yes" or check == "Yes" or check == "YES" or check == "y":
            print("try another number")
        elif check == "N" or check == "No" or check == "NO" or check == "n" or check == "no":
            print("Guess finished by player.")
            break
        else:
            print("Invalid input")
