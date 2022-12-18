# Top-down view Minecraft

This is my clone of Minecraft, built without a game engine, purely on WGPU (and some egui for interface).

It is a turn-based, 2.5d game: the field looks 2d, but you can actually break the top block, or place a block on top of an existing one.

The grey circle indicates the height of the top block (the one you see). The darker the circle, the lower the block.
<img width="51" alt="image" src="https://user-images.githubusercontent.com/27450370/198854409-9959c002-301e-4873-b9f6-7f2131c82fad.png">

Key features:
 - The game has crafting, and for some items you will need a crafting table nearby.
 - The map infinite in 2 dimensions.
 - Your attack damage depends on the items in your inventory.
 - Whether you can break some blocks depends on the items in your inventory.
 - Zombies on unloaded chunks do not vanish. They remain in place until the chunk is loaded again.
 - Banelings (hostile mobs) are better than creepers: they blow up and destroy walls if they can't find a route to player, even if the player is a few blocks away from the explosion radius.
 - Shot arrows can be reused if they don't break.

### AI-friendly
Part of the purpose of this project is to build a training environment for ML agents.

Such evironment would be similar to the famous [MineRL](https://minerl.io/), as it could have the same complexity in terms of progression: 
mine tree -> make crafting table -> make pickaxe -> etc. 

But would require no computer vision, and would run significantly faster than the original.

## Gameplay screenshots:
<img width="800" alt="image" src="https://user-images.githubusercontent.com/27450370/205680241-c074a1d6-f313-4c28-b18d-93cf71e76321.png">

<img width="802" alt="image" src="https://user-images.githubusercontent.com/27450370/208272156-e5e712cf-3fe6-4f8d-afff-71f8c5db42aa.png">

<img width="800" alt="image" src="https://user-images.githubusercontent.com/27450370/198876634-ac05daad-5f30-4b95-ad6b-970feb02e8a1.png">

<img width="801" alt="image" src="https://user-images.githubusercontent.com/27450370/198854546-10c64335-58cb-4a22-93f7-06bdbebad75b.png">

