from pycpsp.bgmodel.config import getDefaults

models = [
    {
        'name': 'birds',
        'requiredKeys': ['energy'],
        'bands': [
            ('energy', 109, 90)
        ],
        'bgmodels': {
            'energy': getDefaults(0.5, subtract=0.1)
        }
    },
    {
        'name': 'car',
        'requiredKeys': ['tone'],
        'bands': [
            ('tone', 40, 21),
        ],
        'bgmodels': {
            'tone': getDefaults(8, subtract=4)
        }
    }
]
"""
    {
        'name': 'speech',
        'requiredKeys': ['tone'],
        'bands': [
            ('tone', 10, 60)
        ],
        'bgmodels': {
            'tone': getDefaults(1.5, subtract='raw', mask=[-10,20])
        }
    },
    {
        'name': 'bus',
        'requiredKeys': ['noise'],
        'bands': [
            ('noise', 10, 60)
        ],
        'bgmodels': {
            'noise': getDefaults(15, subtract='raw', mask=[-10,20])
        }
    },
    {
        'name': 'scooter',
        'requiredKeys': ['tone'],
        'bands': [
            ('noise', 30, 60)
        ],
        'bgmodels': {
            'noise': getDefaults(14, subtract='raw', mask=[-10,20])
        }
    },
]"""
