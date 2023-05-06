from nerf.provider import NeRFDataset
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="datasets dir")
    
    opt = parser.parse_args()

    data = NeRFDataset(opt, device='cpu', type="all")
