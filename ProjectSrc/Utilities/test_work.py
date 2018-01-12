import json
path = '/Users/royhirsch/Documents/BioLog_301117/BioLog_301117_parsed_30112017_111938/00026-1512033102657-match.seq/00026_match_00000_facedet_NA_Flood_probeNone-MetadataMfile.json'
f = open(path, 'rb')
out = json.load(f)
f.close()
# 
# def byteify(input):
#     if isinstance(input, dict):
#         return {byteify(key): byteify(value)
#                 for key, value in input.iteritems()}
#     elif isinstance(input, list):
#         return [byteify(element) for element in input]
#     elif isinstance(input, unicode):
#         return input.encode('utf-8')
#     else:
#         return input
# 
# out2 = byteify(out)
# print(out2)

