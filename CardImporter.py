import StorageUtils as su

storage = su.Storage_struct()

storage.import_images('C:\\Program Files (x86)\\Gatherer Extractor\\pics')
#storage.import_images('C:\\Users\\J\\Desktop\\cardreader2')

storage.export_dict("C:\\Users\\J\\Desktop\\cardreader2\\dict.pickle")