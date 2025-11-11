class School():
  def __init__(self, name, level, numberOfStudents):
    self.name = name
    self.level = level
    self.numberOfStudents = numberOfStudents

  def get_name(self):
    return self.name

  def get_level(self):
    return self.level

  def get_numberOfStudents(self):
    return self.numberOfStudents
  
  def set_numberOfStudents(self, setNumberOfStudents):
    self.numberOfStudents = setNumberOfStudents

  def __repr__(self):
    return f"A {self.level} school named {self.name} with {self.numberOfStudents} students. "

class Primary(School):
  level = "primary"
  def __init__(self, name, numberOfStudents, pickupPolicy):
    super().__init__(name, Primary.level, numberOfStudents)
    self.pickupPolicy = pickupPolicy

  def set_pickupPolicy(self, pickupPolicy):
    self.pickupPolicy = pickupPolicy

  def get_pickupPolicy(self):
    return self.pickupPolicy

  def __repr__(self):
    parentRepr = super().__repr__()
    return parentRepr + "The pickup policy is {pickupPolicy}".format(pickupPolicy = self.pickupPolicy)

class Middle():
  pass

class High(School):
  def __init__(self, name, numberOfStudents, sportsTeams):
    super().__init__(name, 'high', numberOfStudents)
    self.sportsTeams = sportsTeams

  def get_sportsTeams(self):
    return self.sportsTeams

  def set_sportsTeams(self, sportsTeams):
    self.sportsTeams = sportsTeams

  def __repr__(self):
    parentRepr = super().__repr__()
    return parentRepr + "The sports teams are {sportsTeams}".format(sportsTeams = self.sportsTeams)


a = School('Code', 'high', 100)

print(a)
print(a.get_name())
print(a.get_level())
a.set_numberOfStudents(200)
print(a.get_numberOfStudents())

b = Primary('BetterYou', 300, "Pickup Allowed")

print(b.get_pickupPolicy())
print(b)

c = High("CodeBest", 500, ["Tennis", "Basketball"])
print(c.get_sportsTeams())
print(c)
