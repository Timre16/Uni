#Script for converting

#function for converting to binary number into a decimal one
def bin_to_decimal_fraction(binary_str, weights):
    """Converts a binary number into a decimal number while using the prededined weights for each bit."""
    decimal_value = 0.0
    for bit, weight in zip(binary_str, weights):
        if bit == '1':
            decimal_value += weight
    return decimal_value

#function for multiplying two binary numbers in decimal
def multiply_custom_binary(bin1, bin2):
    """Multiplies two binary numbers with specific number formats"""
    # define the weights for the input numbers from 1 to 1/64
    weights_bin1 = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]  # Y,XXXXXX

    # define the weights for the binary system
    weights_bin2 = [8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625]  # 8,4,2,1,0.5,0.25,0.125,0.0625

    # converts the binary numbers into their decimal equivalents
    decimal1 = bin_to_decimal_fraction(bin1, weights_bin1)
    decimal2 = bin_to_decimal_fraction(bin2, weights_bin2)

    # multiplies the two values
    result = decimal1 * decimal2
    return result





###############################
###         main            ###
###############################

bin1 = input("Please enter the first 7-bit binary number for the value X in the format YXXXXXX: ")
bin2 = input("Please enter the second 8-bit number for the weigh Y: ")

result = multiply_custom_binary(bin1, bin2)
print(f"The result in decimal is: {result}")
