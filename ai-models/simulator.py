class artillery_simulator:
    """
    """
    time = 0
    ArtilleyList = []
    TargetAreaDict = {}
    RateOfFire_PerSec = {}

    def __init__(self):
        pass

    ################### Artilley Class ###################
    class Artilley:
        """
        """
        Name = ""
        EnemyInTargetArea = []
        RateOfFire_PerSec = 0
        def __init__(self, Name :str, EnemyInTargetArea :list[str], RateOfFire :float):
            self.Name = Name
            self.EnemyInTargetArea = EnemyInTargetArea
            self.RateOfFire_PerSec = RateOfFire

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
        
        AllEnemy = []

        while AllEnemy != []:
            pass

            #self.getClosestTIme()
            



    class set_area:
        def __init__(self):
            pass


            






def main():
    ################### Example Use ###################

    # Declare Simulator Object
    bf1 = artillery_simulator()

    # Add Artilley Obj to Simulator
    bf1.addNewArtilley("BigA", ["1", "2" ,"3"])
    bf1.addNewArtilley("BigB", ["3", "4" ,"5"])

    # Get List of Artilley in Simulator
    print(bf1.getArtilleyList())

    # Get Enemy in Target Area of Artilley in Simulator
    for i in bf1.getArtilleyList():
        print(i , bf1.getEnemy(i))

    # Run Simulator
    bf1.RunSimulator()

if __name__ == "__main__":
    main()















