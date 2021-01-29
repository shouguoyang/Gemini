
# 1 Train the model and get model saved

# 2 Get features of function

## preparation

1. IDA Pro 7.x
2. set `path of idat` to idapath variable in script `extract_feature_main.py` of line 5.

## 2.1 extract vulnerability function feature
`python extract_feature_main.py -o str_free.json -f str_free nbsmtp`

## 2.2 extract all target function feature
`python extract_feature_main.py -o nbsmtp.json  nbsmtp`

Notice : extract features of all function without `-f func_name` 

# 3 Calculate the similarities between vulnerability and target functions

`python similarity.py str_free.json nbsmtp.json`

the result save in `str_free_nbsmtp.res`

```angular2
[[-1.0, "str_free"], 
[-0.985859751701355, "__do_global_dtors_aux"], 
[-0.8872121572494507, "deregister_tm_clones"], 
[-0.8532835245132446, "register_tm_clones"]]
```