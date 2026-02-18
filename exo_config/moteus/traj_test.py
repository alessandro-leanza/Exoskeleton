#!/usr/bin/python3 -B

import asyncio
import moteus
import time


'''Demonstrates how to specify alternate registers to query, and how
to control the velocity and acceleration limits on a per-command basis
to create a continouous trajectory.'''


async def main():
    qr = moteus.QueryResolution()
    qr._extra = {
        moteus.Register.CONTROL_POSITION : moteus.F32,
        moteus.Register.CONTROL_VELOCITY : moteus.F32,
        moteus.Register.CONTROL_TORQUE : moteus.F32,
        moteus.Register.POSITION_ERROR : moteus.F32,
        moteus.Register.VELOCITY_ERROR : moteus.F32,
        moteus.Register.TORQUE_ERROR : moteus.F32,
        }

    transport = moteus.Fdcanusb()
    c = moteus.Controller(id=1, transport=transport)

    await c.set_stop()

    while True:
        current_command = 1.5 if (round(time.time()) % 2) else -1.5
        results = await c.set_position(
            position=current_command,
            velocity=0.0,
            accel_limit=8.0,
            velocity_limit=3.0,
            query=qr,
        )

        print(results)

        await asyncio.sleep(0.001)


if __name__ == '__main__':
    asyncio.run(main())