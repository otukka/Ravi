# -*- coding: utf-8 -*-

import pathlib
import gzip
import json

class gzipWrapper():
    
    def __init__(self):
        self.internal_ = []
    
    def add(self, jsonObject):
        self.internal_.append(jsonObject)
    
    def save(self, path, content=None):

        if content == None:
            content = self.internal_

        assert(path.suffix == '.gz')
        
        with gzip.open(path, 'wt',  encoding='utf-8') as f:
            json.dump(content, f)

            
    def read(self, path):        
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)

    
    # alternative ways:
        
    # def save(self, path, content=None):

    #     if content == None:
    #         content = self.internal_

    #     assert(path.suffix == '.gz')
        
        
    #     json_str = json.dumps(content) + "\n"
    #     json_bytes = json_str.encode('utf-8')
        
    #     with gzip.GzipFile(path, 'wb') as f:
    #         f.write(json_bytes)               


    # def read(self, path):        
    #     with gzip.GzipFile(path, 'rb') as f:
    #         json_bytes = f.read()
        
    #     json_str = json_bytes.decode('utf-8')
    #     return json.loads(json_str)


def test():
    
    def get_file(path):
        with open(path, 'rb') as f:
            return json.load(f)
        
    p1 = pathlib.Path('D:/Ravi_json/raw_json_old/race__results/2019-01-11 T21 4513430.json')
    p2 = pathlib.Path('D:/Ravi_json/raw_json_old/race__results/2019-01-11 T21 4513437.json')
    
    obj1 = get_file(p1)
    obj2 = get_file(p2)
    
    
    gW = gzipWrapper()
    gW.add(obj1)
    gW.add(obj2)
    

    save_p = pathlib.Path('D:/Ravi_json/asdf.gz')
    gW.save(save_p)
    
    recovered = gW.read(save_p)    

    obj1_rec = recovered[0]    
    obj2_rec = recovered[1]
    
    assert obj1 == obj1_rec
    assert obj2 == obj2_rec  
    
    assert obj1 != obj2
    assert obj1 != obj2_rec
    assert obj2 != obj1_rec


def test():
    
    def get_file(path):
        with open(path, 'rb') as f:
            return json.load(f)
        
    p1 = pathlib.Path('D:/Ravi_json/raw_json_old/race__results/2019-01-11 T21 4513430.json')
    p2 = pathlib.Path('D:/Ravi_json/raw_json_old/race__results/2019-01-11 T21 4513437.json')
    
    obj1 = get_file(p1)
    obj2 = get_file(p2)
    
    
    gW = gzipWrapper()

    

    save_p = pathlib.Path('D:/Ravi_json/asdf.gz')
    gW.save(save_p, [obj1, obj2])
    
    recovered = gW.read(save_p)    

    obj1_rec = recovered[0]    
    obj2_rec = recovered[1]
    
    assert obj1 == obj1_rec
    assert obj2 == obj2_rec  
    
    assert obj1 != obj2
    assert obj1 != obj2_rec
    assert obj2 != obj1_rec

def test():
    

    
    obj1 = "{'asdf':'qwerty'}"
    obj2 = "['zzzzzz']"
    
    
    gW = gzipWrapper()

    

    save_p = pathlib.Path('D:/Ravi_json/asdf.gz')
    gW.save(save_p, [obj1, obj2])
    
    recovered = gW.read(save_p)    

    obj1_rec = recovered[0]    
    obj2_rec = recovered[1]
    
    assert obj1 == obj1_rec
    assert obj2 == obj2_rec  
    
    assert obj1 != obj2
    assert obj1 != obj2_rec
    assert obj2 != obj1_rec

def test():
    

    
    obj1 = str('{"asdf":"qwerty"}')
    obj2 = str('["zzzzzz"]')
    

    
    gW = gzipWrapper()
    gW.add(obj1)
    gW.add(obj2)
    

    save_p = pathlib.Path('D:/Ravi_json/asdf.gz')
    gW.save(save_p)
    
    recovered = gW.read(save_p)    

    obj1_rec = recovered[0]    
    obj2_rec = recovered[1]
    
    assert obj1 == obj1_rec
    assert obj2 == obj2_rec  
    
    assert obj1 != obj2
    assert obj1 != obj2_rec
    assert obj2 != obj1_rec


def test():
    a = [1,'asdf',None]
    b = json.dumps(a)
    c = json.loads(b)
    assert a == c
    
    