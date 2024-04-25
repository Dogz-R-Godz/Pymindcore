import os
import sys
import json
from .errors import *

def injectPlugins(network, pluginDirectory="Plugins"):
    current_directory = os.getcwd()
    print(current_directory)
    pluginsDir=current_directory
    sys.path.insert(0, pluginsDir)
    print(sys.path)
    with open(f"{pluginDirectory}\plugin.json", "r") as f:
        plugin_config=json.load(f)

    pluginName=plugin_config["name"]
    pluginVersion=plugin_config["version"]
    pluginPmcversion=plugin_config["pmcversion"]
    pluginCreator=plugin_config["creator"]
    pluginDescription=plugin_config["description"]

    pluginFname=plugin_config["fname"]

    pluginFunctions=plugin_config["functions"]
    pluginInitfunc=plugin_config["initfunc"]
    pluginImports=plugin_config["imports"]
    pluginCvars=plugin_config["cvars"]


    plugin=__import__("Plugins." + pluginFname).__dict__[pluginFname]
    print("Successfully imported plugin file")


    pluginDict=plugin.__dict__

    if network.version>pluginPmcversion:
        print(f"Warning: outdated plugin. Plugin version {pluginVersion} vs Pymindcore version {network.version}")
    elif network.version<pluginPmcversion:
        print(f"Warning: plugin from the future. Please update Pymindcore to version {pluginPmcversion} (currently {network.version})")



    functions=pluginFunctions
    print("Successfully found the functions")

    for name in functions: # iterate through every module's attributes
        try:
            val=pluginDict[name]
            try:
                setattr(network, name, val.__get__(network, network.__class__))
            except:
                raise Exception("FunctionNotFunctionError")
        except KeyError as e:
            raise Exception("FunctionNotFoundError") from e
    print("Successfully added the functions")

    cvars=pluginCvars

    for name in cvars:
        try:
            val=pluginDict[name]
            try:
                setattr(network, name, val)
            except:
                raise Exception("VariableNotVariableError")
        except KeyError as e:
            raise Exception("VariableNotFoundError") from e
    print("Successfully added the class variables")

    for imp in pluginImports:
        globals()[pluginImports[imp]]=__import__(imp)
        print(f"Successfully imported {imp} as {pluginImports[imp]}")
    print("Successfully imported all imports")

    initFunc=getattr(network, pluginInitfunc)

    initFunc(network.pluginSettings)
    print("Successfully initialised plugin")




    print(f"Successfully loaded plugin '{pluginName}' by {pluginCreator}")
    print("Description: ")
    for line in pluginDescription:
        print(line)
    x=0
    return network
