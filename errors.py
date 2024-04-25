#All the errors: 

class invalidActivationError(Exception):
    def __init__(self, message=None):
        self.message=message
        print("Activation does not match any default activations")
        super().__init__(message)

class invalidOutputActivationError(Exception):
    def __init__(self, message=None):
        self.message=message
        print("Output activation does not match any default activations")
        super().__init__(message)

class asianParentsExpectationsError(Exception):
    def __init__(self, message=None):
        self.message=message
        print("Stop expecting too much from your FAILURE... (actually tho, dont make the expected output higher than is possible from that activation. )")
        super().__init__(message)

class howDoYouFuckUpSavingError(Exception):
    def __init__(self, message=None):
        self.message=message
        print("How did you fuck up the saving? Thats impressive. *cries in getting this error over 30 times in development*")
        super().__init__(message)
    
class corruptedOrOldSaveFileError(Exception):
    def __init__(self, message=None):
        self.message=message
        print("Save file is out of date or corrupted.")
        super().__init__(message)

class invalidSaveNetworkShapeError(Exception):
    def __init__(self, message=None):
        self.message=message
        print("Invalid network shape for saving")
        super().__init__(message)

class inhomogenousAhhhhhhhError(Exception):
    def __init__(self, message=None):
        self.message=message
        print("This error has plagued me for ages. If you get this error, and you're not using any plugins, then report it in the discord, and I'll try and fix it.")
        super().__init__(message)

class invalidOrMissingNetworkStateError(Exception):
    def __init__(self, message=None):
        self.message=message
        print("The network state is missing. Please make sure that network.a is a list with the state of the network in it.")
        super().__init__(message)

class networkWentToGetMilkError(Exception):
    def __init__(self, message=None):
        self.message=message
        print("The network itself is missing. Must have needed alot of milk...")
        super().__init__(message)
        

"""  
Base pymindcore errors:
    invalidActivationError
    invalidOutputActivationError
    asianParentsExpectationsError
    howDoYouFuckUpSavingError
    corruptedOrOldSaveFileError
    invalidSaveNetworkShapeError
    inhomogenousAhhhhhhhError
    invalidOrMissingNetworkStateError
    networkWentToGetMilkError

Plugin errors:
    pluginJsonNotFoundError
    pluginJsonIncorrectTypeError
    pluginJsonNameNotFoundError
    pluginJsonVersionNotFoundError
    pluginJsonPmcversionNotFoundError
    pluginJsonCreatorNotFoundError
    pluginJsonDescriptionNotFoundError
    pluginJsonFnameNotFoundError
    pluginFileNotFoundError
    pluginJsonFunctionsNotFoundError
    pluginFunctionsNotFoundError
    pluginJsonImportsNotFoundError
    pluginImportsNotFoundError
    pluginModuleNotFoundError
    pluginJsonSettingsNotFoundError 
    pluginJsonCvarsNotFoundError


    """

