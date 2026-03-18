This is a callback pattern I implemented in a game so that the sprite animation manager doesn't constantly have to loop and check which entities are dead or alive. Dying entities automatically broadcast their death event to the manager and the manager know how to animate the event accordingly.

To formulate the problem, we have two classes -- the `Entity` (it can be an enemy class). and the `Manager` (sprite animation manager). The goal is that when any entity calls `Die`, the sprite animation manager must play an animation at the death cell.

To do this, the entity stores a public method `OnDie()` as an `std::function` that can be set to a custom function in the manager.
This is done by the setter `SetOnDie()` in the code and for every entity, the manager calls `SetOnDie` to overwrite each entity's `OnDie` method. This is illustrated in step **(1)** of the diagram, and in the following lines of code:
```cpp
// in Entity:
using OnDieFn = std::function<void(EntityType, Cell)>;
OnDieFn on_die_;
void SetOnDie(OnDieFn cb) { on_die_ = std::move(cb); }

// in manager:
public:
  // injects its own functionality into OnDie of Entity
  void ListenOnDie(Entity &e) {
    auto fn = [this](EntityType t, Cell c) { DieAnimation(t, c); };
    e.SetOnDie(fn);
  }

private:
  void DieAnimation(EntityType t, Cell c) const {
     // do stuff
  }
```
That's it! We hooked the managed into each entity.

The next is easier and it involves the implementation of each entity's `Die()` calling `on_die_()`, given that the latter is already referring to the sprite manager. This is step **(2)** in the diagram and it's  achieved by the following lines:

```cpp
  // internal stuff and shared (on_die_) functionality
  virtual void Die() {
    // Do stuff and figure out if we're dead
    // Call the functionality shared with manager (listener)
    if (on_die_) on_die_(type_, cell_);
  }
```

```
Entity                                   Manager
+-------------------------+              +-------------------------------+
|   // meant to be shared |    (1)       |                               |
| + fn OnDie()            |<-------------| + fn ListenForDie(Entity e) { |
|                         |              |     // assignment - not call  |
|                         |              |     e.OnDie = RunOnDie        |
| + fn Die() {            |              |   }                           |
|     // do stuff         |    (2)       |                               |
|     OnDie()             |-----+        |                               |
|   }                     |     +------->| + fn RunOnDie() {             |
|                         |              |     // do additional stuff    |
|                         |              |   }                           |
|                         |              |                               |
+-------------------------+              +-------------------------------+

(1) for each Entity e: entities:
      manager.ListenForDie(e)
(2) for each Entity e: entities:
      e.Die()
```

The full code demonstrating the callback mechanism is listed below.

```cpp
#include <iostream>
#include <functional>
#include <memory>
#include <vector>

enum class EntityType { Generic, Player, Enemy };

struct Cell { int x = 0; int y = 0; };

class Entity {
public:
  using OnDieFn = std::function<void(EntityType, Cell)>;
  Entity(int id, EntityType type, Cell xy)
    : id_(id), type_(type), cell_(xy), alive_(true) {}
  virtual ~Entity() = default;
  void SetOnDie(OnDieFn cb) { on_die_ = std::move(cb); }

  // internal stuff and shared (on_die_) functionality
  virtual void Die() {
    if (!alive_) return;
    alive_ = false;
    // call the functionality shared with manager (listener)
    if (on_die_) on_die_(type_, cell_);
  }

  int id() const { return id_; }
  EntityType type() const { return type_; }

protected:
  int id_;
  EntityType type_;
  Cell cell_;
  bool alive_;
  // meant to be shared/written by the manager
  OnDieFn on_die_;
};

class SpriteManager {
public:
  // injects its own functionality into OnDie of Entity
  void ListenOnDie(Entity &e) {
    auto fn = [this](EntityType t, Cell c) { DieAnimation(t, c); };
    e.SetOnDie(fn);
  }

private:
  void DieAnimation(EntityType t, Cell c) const {
    std::cout << "[SpriteManager] DieAnimation for type="
              << static_cast<int>(t) << " at cell=("
              << c.x << "," << c.y << ")\n";
  }
};

std::unique_ptr<Entity> SpawnEntity(int id, EntityType t, Cell cell ) {
  return std::make_unique<Entity>(id, t, cell);
}

int main() {
  SpriteManager mgr;
  std::vector<std::unique_ptr<Entity>> entities;
  entities.push_back(SpawnEntity(1, EntityType::Player, Cell{2, 3}));
  entities.push_back(SpawnEntity(2, EntityType::Enemy, Cell{5, 6}));

  // listen for die event by injecting its functionality in them 
  for (auto &e : entities) {
    mgr.ListenOnDie(*e);
  }
  auto spawned = SpawnEntity(3, EntityType::Enemy, Cell{7, 8});
  mgr.ListenOnDie(*spawned);
  entities.push_back(std::move(spawned));

  entities[1]->Die();
  entities[2]->Die();

  return 0;
}
```
