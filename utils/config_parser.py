import yaml
import os


class ProjectConfig:
    def __init__(self, config_data, file_path=None):
        self.file_path = file_path  # Store the path of the YAML file
        for key, value in config_data.items():
            if isinstance(value, dict):
                setattr(self, key, ProjectConfig(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

        # Function to check for nested attributes
    def has_nested_attribute(self, attr_chain):
        attrs = attr_chain.split('.')
        obj = self
        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return False
        return True
    # Function to add or update an attribute
    def add_attribute(self, key, value):
        if isinstance(value, dict):
            setattr(self, key, ProjectConfig(value))
        else:
            setattr(self, key, value)

    # Function to add an item to a list
    def add_to_list(self, list_name, item):
        current_list = getattr(self, list_name, None)
        if isinstance(current_list, list):
            current_list.append(item)
        else:
            raise TypeError(f"{list_name} is not a list.")

    # Recursive function to convert the class back to a dictionary (for saving to YAML)
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ProjectConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    # Function to save the configuration back to the YAML file
    def save_config(self):
        if self.file_path is None:
            raise ValueError("No file path specified for saving the configuration.")
        with open(self.file_path, 'w') as file:
            yaml.dump(self.to_dict(), file)

    # Function to dynamically access nested attributes
    def get_dynamic_attr(self, attr_chain, dynamic_var):
        """
        Accesses a nested attribute dynamically where part of the chain is variable.

        Parameters:
        - attr_chain: The attribute chain with a placeholder for the dynamic part (e.g., "run_config.{var}.csv")
        - dynamic_var: The variable part that replaces the placeholder

        Returns:
        - The requested attribute value or raises an error if not found
        """
        # Replace the placeholder {var} with the actual dynamic variable
        attr_chain = attr_chain.replace("{var}", dynamic_var)

        # Split the chain into parts to access attributes step by step
        attrs = attr_chain.split('.')
        obj = self
        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                raise AttributeError(f"Attribute '{attr}' not found in the chain '{attr_chain}'.")

        return obj

# Function to load the YAML file and instantiate the class
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config_data = yaml.safe_load(file)
    return ProjectConfig(config_data, file_path=yaml_file)

# def has_nested_attribute(obj, attr_chain):
#     """Check if a nested attribute exists in the form of 'attr1.attr2.attr3'."""
#     attrs = attr_chain.split('.')
#     for attr in attrs:
#         if hasattr(obj, attr):
#             obj = getattr(obj, attr)
#         else:
#             return False
#     return True

# Usage example:
if __name__ == "__main__":
    config = load_config("proj_config.yaml")
    print(config)  # Print the whole configuration as a class

    # Add a new attribute
    config.add_attribute("new_attribute", "new_value")

    # Add a new item to a list (make sure it's a valid list name from the YAML file)
    config.add_to_list("col_var_ids", "new_id")

    # Save the changes back to the file
    config.save_config()
