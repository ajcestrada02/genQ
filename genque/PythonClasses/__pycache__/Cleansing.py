def UniqueItems(Noun):
    seen = set()
    result = []
    for item in Noun:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def removeRegexList(list):
	nstr = ""
	nstr = re.sub(r'[""|,|\\|''|)|(]', r'', str(list))

	return nstr
