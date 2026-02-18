#!/usr/bin/python3

"""
This example commands a single servo at ID #1 using the default
transport to hold the current position indefinitely, and prints the
state of the servo to the console.
"""

import asyncio
import math
import moteus

async def main():

    transport = moteus.Fdcanusb()
    controller = moteus.Controller(id=1, transport=transport)  # Change id if your board uses a different ID
    await controller.set_stop()

    while True:

        # state = await controller.set_position(position=math.nan, query=True)
        state = await controller.query()

        # Print out just the position register.
        print("Position:", state.values[moteus.Register.POSITION])

        await asyncio.sleep(0.05)

if __name__ == '__main__':
    asyncio.run(main())