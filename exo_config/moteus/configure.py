import asyncio
import moteus

async def main():
    """
    Connects to a moteus controller, sets its min/max position limits,
    and saves the configuration permanently.
    """
    # Change this to your controller's ID
    CONTROLLER_ID = 1

    # Define your desired min/max positions in revolutions
    MIN_POS_REV = -0.5
    MAX_POS_REV = 0.5

    print(f"Connecting to controller {CONTROLLER_ID}...")
    # Create a controller instance. The transport will be auto-detected.
    c = moteus.Controller(id=CONTROLLER_ID)

    # Stop the controller to allow configuration changes.
    await c.set_stop()

    print("Setting configuration...")
    # Set the min and max position limits.
    # We use the 'query=False' to indicate we are just sending a command.
    await c.set("servo.position.min", MIN_POS_REV, query=False)
    await c.set("servo.position.max", MAX_POS_REV, query=False)

    print("Saving configuration to permanent memory...")
    # Save the configuration.
    await c.set("conf.save", 1, query=False)

    print("\nConfiguration saved successfully!")
    print(f"  servo.position.min: {MIN_POS_REV}")
    print(f"  servo.position.max: {MAX_POS_REV}")

    # It's good practice to wait a moment for the save to complete
    # before doing anything else, like a reboot.
    await asyncio.sleep(1.0)

    # You can optionally reboot the controller to apply the new settings.
    # print("Rebooting controller...")
    # await c.set("conf.reboot", 1, query=False)


if __name__ == '__main__':
    asyncio.run(main())

