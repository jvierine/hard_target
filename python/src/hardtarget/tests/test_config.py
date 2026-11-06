import unittest, os, io, json, shutil
from ..config import Config


class test_Config(unittest.TestCase):
    def setUp(self):
        
        #Create a test dictionary to supply to config class, and to 
        #compare the json tests against
        self.expected_json_dict = {
            'test_dict' : {'test' : 'test'},
            'test_int' : 2,
            'test_list' : ['test1', 'test2'],
            'test_string' : 'test',
        }

        #Create a directory to compare the ini tests against
        self.expected_ini_dict = {
                #INI files only reads values as strings
                'test_header':{
                    'test_string' : 'test123',
                }
        }
        
        #Create a test json string
        self.test_json_string = \
        """{
            "test_dict" : {
                "test" : "test"
            },
            "test_int" : 2,
            "test_list" : [
                "test1", 
                "test2"
            ],
            "test_string" : "test"
        }"""

        #Create a test INI string and remove whitespace
        self.test_ini_string = \
        """[test_header]\ntest_string = test123\n\n"""

        #Test file names
        self.test_json_file = "/tmp/test_config_json.json"
        self.test_ini_file = "/tmp/test_config_INI.ini"
        self.test_output_dir = "/tmp/test_config_folder"
        #Create test file to read json from
        with open(self.test_json_file, 'w') as test_jsonfile:
            test_jsonfile.write(self.test_json_string)

        #Create test file to read ini from
        with open(self.test_ini_file, 'w') as test_inifile:
            test_inifile.write(self.test_ini_string)     

    def test_config_as_dict(self):

        #Create config object
        c = Config.from_dict(self.expected_json_dict)

        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_int'], self.expected_json_dict['test_int'])
        self.assertEqual(c['test_string'], self.expected_json_dict['test_string'])
        self.assertEqual(c['test_list'], self.expected_json_dict['test_list'])
        self.assertEqual(c['test_dict'], self.expected_json_dict['test_dict'])
    
    def test_config_as_json_string(self):

        #Create config object
        c = Config.from_string(self.test_json_string)

        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_int'], self.expected_json_dict['test_int'])
        self.assertEqual(c['test_string'], self.expected_json_dict['test_string'])
        self.assertEqual(c['test_list'], self.expected_json_dict['test_list'])
        self.assertEqual(c['test_dict'], self.expected_json_dict['test_dict'])
        self.assertFalse(c.values_as_strings)

    def test_config_as_ini_string(self):

        #Create config object
        c = Config.from_string(self.test_ini_string)
        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_header']['test_string'], self.expected_ini_dict['test_header']['test_string'])
        self.assertTrue(c.values_as_strings)

    def test_config_as_json_file(self):
        
        #Create config object
        c = Config.from_file(self.test_json_file)

        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_int'], self.expected_json_dict['test_int'])
        self.assertEqual(c['test_string'], self.expected_json_dict['test_string'])
        self.assertEqual(c['test_list'], self.expected_json_dict['test_list'])
        self.assertEqual(c['test_dict'], self.expected_json_dict['test_dict'])
        self.assertFalse(c.values_as_strings)


    def test_config_as_ini_file(self):

        #Create config object
        c = Config.from_file(self.test_ini_file)

        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_header']['test_string'], self.expected_ini_dict['test_header']['test_string'])
        self.assertTrue(c.values_as_strings)

    def test_config_as_json_stream(self):
        with open(self.test_json_file) as file:
            #Create config object
            c = Config.from_stream(file)

            #Assert that that the config object equalls the test dictionary
            self.assertEqual(c['test_int'], self.expected_json_dict['test_int'])
            self.assertEqual(c['test_string'], self.expected_json_dict['test_string'])
            self.assertEqual(c['test_list'], self.expected_json_dict['test_list'])
            self.assertEqual(c['test_dict'], self.expected_json_dict['test_dict'])
            self.assertFalse(c.values_as_strings)


    def test_config_as_ini_stream(self):
        with open(self.test_ini_file, 'r') as file:
            #Create config object
            c = Config.from_stream(file)

            #Assert that that the config object equalls the test dictionary
            self.assertEqual(c['test_header']['test_string'], self.expected_ini_dict['test_header']['test_string'])
            self.assertTrue(c.values_as_strings)

    def test_change_config_value(self):
        test_new_int = 5
        #Create config object
        c = Config(self.expected_json_dict)
        
        c.set_param('test_int', test_new_int)
        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_int'], test_new_int)
        self.assertEqual(c['test_string'], self.expected_json_dict['test_string'])
        self.assertEqual(c['test_list'], self.expected_json_dict['test_list'])
        self.assertEqual(c['test_dict'], self.expected_json_dict['test_dict'])
    
    def test_write_config_to_json_file(self):
        #Create config object based on json dict
        c = Config(self.expected_json_dict)
        #Save paramaters to json file
        c.save_param('test', output_dir=self.test_output_dir)

        #Create expected json string from json dict
        expectedString = json.dumps(self.expected_json_dict, sort_keys=True, indent=4)
        expectedString = io.StringIO(expectedString)

        #Check generated file against expected string
        with open(self.test_output_dir + '/test.json') as file:
            for line in file:
                expectedLine = expectedString.readline()
                self.assertEqual(line, expectedLine, f'Expected "{expectedLine}" got {line}')
        
    def test_write_config_to_ini_file(self):
        c = Config(self.expected_ini_dict)
        c.save_param('test', output_dir=self.test_output_dir, ini=True)

        teststring = io.StringIO(self.test_ini_string)
        with open(self.test_output_dir + '/test.ini') as file:
            for line in file:
                expectedLine = teststring.readline()
                self.assertEqual(line, expectedLine, f'Expected "{expectedLine}" got {line}')
    
    def test_get_keys(self):
        #Create config object
        c = Config(self.expected_json_dict)

        #Get list of keys
        keys = c.get_keys()
        #Assert that the keys are as expected
        self.assertEqual(keys, list(self.expected_json_dict.keys()))        

    def test_error_when_using_default_values_without_implementing_default(self):
        #Ensure that not implemented error is raised
        with self.assertRaises(NotImplementedError):
            c = Config.from_default()
    
    def test_config_as_default_values(self):
        #Create test class with default values
        class Default_test(Config):
            @classmethod
            def get_default(cls) -> dict:
                return self.expected_json_dict
        
        #Create instance of default test from default values
        c = Default_test.from_default()
        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_int'], self.expected_json_dict['test_int'])
        self.assertEqual(c['test_string'], self.expected_json_dict['test_string'])
        self.assertEqual(c['test_list'], self.expected_json_dict['test_list'])
        self.assertEqual(c['test_dict'], self.expected_json_dict['test_dict'])
    
    def test_config_as_dict_with_default_values(self):
        #Create test class with default values
        class Default_test(Config):
            @classmethod
            def get_default(cls) -> dict:
                return self.expected_json_dict
        #Create expected dict
        expectedDict = {'test_int': 500}
        #Create instance of default test from default values
        c = Default_test.from_dict(expectedDict, True)
        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_int'], expectedDict['test_int'])
        self.assertEqual(c['test_string'], self.expected_json_dict['test_string'])
        self.assertEqual(c['test_list'], self.expected_json_dict['test_list'])
        self.assertEqual(c['test_dict'], self.expected_json_dict['test_dict'])
    
    def test_config_as_json_string_with_default_values(self):
        #Create test class with default values
        class Default_test(Config):
            @classmethod
            def get_default(cls) -> dict:
                newDict = self.expected_json_dict.copy()
                newDict['test_int'] = 500
                return newDict

        #Create instance of default test from default values
        c = Default_test.from_string(self.test_json_string, True)
        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_int'], self.expected_json_dict['test_int'])
        self.assertEqual(c['test_string'], self.expected_json_dict['test_string'])
        self.assertEqual(c['test_list'], self.expected_json_dict['test_list'])
        self.assertEqual(c['test_dict'], self.expected_json_dict['test_dict'])
    
    def test_config_as_ini_string_with_default_values(self):
        default_string = '123test'
        #Create test class with default values
        class Default_test(Config):
            @classmethod
            def get_default(cls) -> dict:
                newDict = self.expected_ini_dict.copy()
                newDict['test_header']['test_string'] = default_string
                newDict['test_header']['test_string2'] = default_string
                return newDict
        #Create instance of default test from default values
        c = Default_test.from_string(self.test_ini_string, True)
        #Assert that that the config object equalls the test dictionary
        self.assertEqual(c['test_header']['test_string'], self.expected_ini_dict['test_header']['test_string'])
        self.assertEqual(c['test_header']['test_string2'], default_string)

    def tearDown(self):
        #remove files from setup
        os.remove(self.test_json_file)
        os.remove(self.test_ini_file)
        #remove test output dir folder if created
        try:
            shutil.rmtree(self.test_output_dir)
        except FileNotFoundError:
            #If the file does not exist then the tests will notify about any errors
            pass

if __name__ == '__main__':
    unittest.main()