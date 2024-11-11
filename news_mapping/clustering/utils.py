def create_reverse_mapping(mapping_dict):
    reverse_mapping = {}
    for key, values in mapping_dict.items():
        for value in values:
            reverse_mapping[value] = key
    return reverse_mapping

def replace_values_from_dict(df, column, mapping_dict):
    reverse_mapping = create_reverse_mapping(mapping_dict)
    df[column] = df[column].apply(lambda x: reverse_mapping.get(x, x))
    return df
