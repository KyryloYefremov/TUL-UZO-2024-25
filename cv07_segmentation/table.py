class Table:

    def __init__(self):
        self.lists = []

    def update_table(self, not_zero_neighbours: list, debug=False):
        """
        Update the table of lists with the new values.
        """
        if debug:
            print(f"\tlists: {self.lists}")
            print(f"\tnot_zero_neighbours: {not_zero_neighbours}\n")

        matched_lists = []
        
        # find all lists that contain at least one value from neighbours
        for l in self.lists:
            if any(value in l for value in not_zero_neighbours):
                matched_lists.append(l)

        # if no list contains these values, create a new sublist
        if not matched_lists:
            self.lists.append(list(not_zero_neighbours))
            return

        # merge all lists together
        merged_list = set()
        for l in matched_lists:
            merged_list.update(l)
            self.lists.remove(l)  # remove merged lists from the original list
        
        merged_list.update(not_zero_neighbours)  # add new neighbours
        self.lists.append(list(merged_list))  # add merged list to the orig list

    def get_value_by(self, value, by_fun):
        """
        Get the value from the table by the given function.
        """
        for l in self.lists:
            if value in l:
                return by_fun(l)
        return value


    def __str__(self):
        return str(self.lists)
    
    def __repr__(self):
        return str(self.lists)

                

        
        
        