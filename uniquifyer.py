org_data_file = open("/Users/markus/workspace/master/Master/data/datasets/10_Flickr30k.txt", 'r')
org_lines = org_data_file.readlines()
org_data_file.close()

unique_lines = []
for line in org_lines:
	if line not in unique_lines:
		unique_lines.append(line)

print len(org_lines)
print len(unique_lines)
unique_lines.sort()

uniq_data_file = open("/Users/markus/workspace/master/Master/data/datasets/10_Flickr30k_uniq.txt", 'w+')
uniq_data_file.writelines(unique_lines)
uniq_data_file.close()
