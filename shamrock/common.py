

# ======================================================================================================================
# -- UTILITIES
# ======================================================================================================================
def errmsg(fileName, functionName=None, className=None, methodName=None, message=''):
    """
    Returns a string that contains a message/warning/error along with information on where message originated.

    :param fileName: The file name that contains the code that issued the message.
    :param functionName: Name of the function that issued the message.
    :param className: Name of the class that issued the message.
    :param methodName: Name of the method that issued the message.
    :param message: The message
    :return:
    """

    if functionName is None:
        return f'\n[FILE:{fileName}]\n[CLASS:{className}]\n[METHOD:{methodName}]\n[MESSAGE:{message}]'
    else:
        return f'\n[FILE:{fileName}]\n[FUNCTION:{functionName}]\n[MESSAGE:{message}]'