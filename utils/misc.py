def extract_positions_from_string(data_str):
    quoted_strings = [item.strip('"') for item in data_str.split('"') if item.strip()]
    initial_position = quoted_strings[3]
    final_position = quoted_strings[7]
    return initial_position, final_position

def generate_navigation(tree, start_room, end_room):
    print("Hello sir")
    # Helper function to find the full path of a room
    def find_room_path(current_path, subtree, room_name):
        for key, value in subtree.items():
            if key == room_name:
                return current_path + [key]
            elif isinstance(value, dict):
                result = find_room_path(current_path + [key], value, room_name)
                if result:
                    return result
        return None

    # Function to determine the central node dynamically
    def find_central_node(tree):
        for key, value in tree.items():
            if isinstance(value, dict) and "In Front" in value:
                return key
        return None

    # Function to determine directions between rooms
    def get_directions(tree, start_room, end_room):
        central_node = find_central_node(tree)
        print(central_node)
        if not central_node:
            return "Central point not found in the hierarchy."

        start_path = find_room_path([], tree, start_room)
        end_path = find_room_path([], tree, end_room)

        if not start_path or not end_path:
            return f"Invalid rooms: '{start_room}' or '{end_room}' not found."

        directions = []

        # Handle movement from or to a room in the central point section
        central_section = tree[central_node].get("In Front", {})
        print(central_section)
        if start_room in central_section:
            if "Left Turn" in end_path:
                directions.append("Turn into the left corridor")
                directions.append(f"{end_room} is {tree[central_node]['Left Turn'][end_room]}.")
            elif "Right Turn" in end_path:
                directions.append("Turn into the right corridor")
                directions.append(f"{end_room} is {tree[central_node]['Right Turn'][end_room]}.")
            return " ".join(directions)

        if end_room in central_section:
            if "Left Turn" in start_path:
                directions.append(f"Exit the left corridor and go to the {central_node}")
            elif "Right Turn" in start_path:
                directions.append(f"Exit the right corridor and go to the {central_node}")
            return " ".join(directions)

        if end_room == central_node:
            if "Left Turn" in start_path:
                directions.append(f"Exit the left corridor and go to the {central_node}")
            elif "Right Turn" in start_path:
                directions.append(f"Exit the right corridor and go to the {central_node}")
            directions.append(f"You have reached {end_room}, which is {central_node}.")
            return " ".join(directions)

        # Handle movement from the central node
        if start_room == central_node:
            print("I am here")
            if "Left Turn" in end_path:
                directions.append(f"Go straight and turn in the left corridor")
                directions.append(f"{end_room} is {tree[central_node]['Left Turn'][end_room]}.")
            elif "Right Turn" in end_path:
                directions.append(f"Go straight and turn in the right corridor")
                directions.append(f"{end_room} is {tree[central_node]['Right Turn'][end_room]}.")
            return " ".join(directions)

        # Handle movement between rooms in different corridors
        if "Left Turn" in start_path and "Right Turn" in end_path:
            directions.append(f"Go to the {central_node} and continue straight into the right corridor")
            directions.append(f"{end_room} is {tree[central_node]['Right Turn'][end_room]}.")
        elif "Right Turn" in start_path and "Left Turn" in end_path:
            directions.append(f"Go to the {central_node} and continue straight into the left corridor")
            directions.append(f"{end_room} is {tree[central_node]['Left Turn'][end_room]}.")

        # Handle movement within the same corridor
        elif start_path[:-1] == end_path[:-1]:
            corridor = "Left Turn" if "Left Turn" in start_path else "Right Turn"
            directions.append(f"Go to {end_room}, which is {tree[central_node][corridor][end_room]}.")

        return " ".join(directions)

    return get_directions(tree, start_room, end_room)
