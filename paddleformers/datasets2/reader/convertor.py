

def erniekit_convertor(data):
    all_data = []
    for item in data:
        each_line = dict()
        each_line["messages"] = []
        if "system" in item:
            each_line["messages"].append({"role": "system", "content": item["system"]})
        for q, a in zip(item["src"], item["tgt"]):
            each_line["messages"].append({"role": "user", "content": q.strip()})
            each_line["messages"].append({"role": "assistant", "content": a.strip()})
        all_data.append(each_line)

    return all_data
        
