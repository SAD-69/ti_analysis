from arcgis.gis import GIS, ContentManager

ANEEL_PORTAL = "https://dadosabertos-aneel.opendata.arcgis.com/"

class Aneel:
    def __init__(self, username: str = None, password: str = None):
        self.gis = self._gis
        self.username = username
        self.password = password
        self.content = self._content

    @property
    def _gis(self):
        return GIS()
    
    @property
    def _content(self) -> ContentManager:
        return self.gis.content
    
    def get_item(self, agent: str, item_type: str = "File Geodatabase"):
        return self.content.search(query=f"title:{agent}", item_type=item_type)
    

if __name__ == '__main__':
    aneel = Aneel()
    item = aneel.get_item("CEEE")
    print(item[0].title)