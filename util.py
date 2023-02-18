import argparse
import yaml

def Recursive_Parse(args_Dict):
    parsed_Dict = {}
    for key, value in args_Dict.items():
        if isinstance(value, dict):
            value = Recursive_Parse(value)
        parsed_Dict[key]= value

    args = argparse.Namespace()
    args.__dict__ = parsed_Dict
    return args
            

if __name__ == "__main__":    
    with open('Hyper_Parameter.yaml') as f:
        hp_Dict = yaml.load(f, Loader=yaml.Loader)

    hp = Recursive_Parse(hp_Dict)


    print(hp.Sound.Spectrogram_Dim)

    args = argparse.Namespace()
    args.epoch = 100

    print(args)