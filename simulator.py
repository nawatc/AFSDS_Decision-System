class artillery_simulator:
    """
    """
    time = 0
    ArtilleyList = []
    TargetAreaDict = {}

    def __init__(self):
        pass

    ################### Artilley Method ###################
    def addNewArtilley(self, ArtilleyName, EnemyInRangeList :list[str]):
        if ArtilleyName not in self.ArtilleyList:
            self.ArtilleyList.append(ArtilleyName)
            self.TargetAreaDict[ArtilleyName] = EnemyInRangeList
        else:
            print("Error: Function - addNewArtilley ", ArtilleyName, " Already in ArtilleyList.")
            return 0
    
    def getArtilleyList(self):
        return self.ArtilleyList
    
    def getEnemyInTargetArea(self, ArtilleyName):
        return self.TargetAreaDict[ArtilleyName]

    ################### Simulator Method ###################
    def getTime(self):
        return self.ArtilleyList
    
    def getChoice(self):
        pass
    
    def getClosestTIme(self):
        pass

    def RunSimulator(self):
        pass


    class set_area:
        def __init__(self):
            pass


            
################### Example Use ###################
bf1 = artillery_simulator()

bf1.addNewArtilley("BigA", ["1", "2" ,"3"])
bf1.addNewArtilley("BigB", ["3", "4" ,"5"])

print(bf1.getArtilleyList())

for i in bf1.getArtilleyList():
    print(bf1.getEnemyInTargetArea(i))





















