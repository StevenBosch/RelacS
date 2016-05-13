import pycpsp.sourceindicators.indicator as indicator
import pycpsp.bgmodel.bgmodel as bgmodel
import pycpsp.bgmodel.config as bgconfig
import pycpsp.plot as plot
import json, copy

"""
    Returns background models, populated model configs, and an indicator.SourceIndicator object for each of the models
    in the config's model definitions
"""
def initFromConfig(definitions, **kwargs):
    keys = []
    bgmodels = {}
    models = []
    for model in copy.deepcopy(definitions):
        keys = keys + model['requiredKeys']
        # hash all bg models, if we have a duplicate don't recreate it.
        # add the bg model to the model dictionary
        for key in model['bgmodels']:
            h = hash(json.dumps(model['bgmodels'][key], sort_keys=True))
            if not h in bgmodels:
                bgmodels[h] = bgmodel.BGModel(h, model['bgmodels'][key], **kwargs)
            #generate subtract model, calculate hash and append if it doesn't exist
            if model['bgmodels'][key]['subtract'] != None:
                subtractmodel = bgconfig.getDefaults(model['bgmodels'][key]['subtract'])
                sh = hash(json.dumps(subtractmodel, sort_keys=True))
                if not sh in bgmodels:
                    bgmodels[sh] = bgmodel.BGModel(sh, subtractmodel, **kwargs)
                bgmodels[h].subtract = sh
            model['bgmodels'][key] = bgmodels[h]
        models.append(model)

    indicators = []
    for model in models:
        indicators.append(indicator.SourceIndicator(model, **kwargs))
    return indicators, models, bgmodels

"""
    Calculates leaky-integrated responses based on indicator definitions
"""
def calculateBGModels(inputs, indicators, bgmodels, normalize=False):
    #transform each key in inputs to a leaky-integrated one
    #keep track of inputs we have calculated
    responses = dict((h, None) for h in bgmodels)
    for i in indicators:
        keys = i.wants()
        for key in keys:
            #bg model hash
            h = i.model['bgmodels'][key].name
            #do we need to calculate the response?
            if responses[h] == None:
                if not key in inputs:
                    raise Exception("Indicator {} wants key {} for comparison, but key is not present in input".format(i.name, key))
                if normalize:
                    inputsignal = plot.imscale(inputs[key], [0,60])
                    response = bgmodels[h].calculate(inputsignal)
                else:
                    response = bgmodels[h].calculate(inputs[key])
                    inputsignal = inputs[key]

                sh = bgmodels[h].subtract
                if sh is not None:
                    if not sh in bgmodels:
                        raise Exception('Trying to subtract from a non-existing model')
                    if responses[sh] == None:
                        responses[sh] = bgmodels[sh].calculate(inputs[key])
                    response = responses[sh] - response
                
                if bgmodels[h].mask is not None:
                    response = plot.imgmask(response, bgmodels[h].mask)
                
                responses[h] = response
            #pass the reference to response to indicator
            i.setBGModel(key, responses[h])
