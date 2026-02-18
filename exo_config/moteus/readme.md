

### python documentation

[https://github.com/mjbots/moteus/tree/main/lib/python](https://github.com/mjbots/moteus/tree/main/lib/python)




### Configure 

python3 -m moteus_gui.tview

conf get id.id
conf set id.id xxx

d stop

conf set servopos.position_min nan
conf set servopos.position_max nan

conf set servo.pid_position.kp 6
conf set servo.pid_position.kd 0.05
conf set servo.pid_position.ki 0.0
conf set servo.pid_position.ilimit 0.0

conf write

https://www.ncnynl.com/docs/en/moteus/reference/



0


### Calibrate motor 
python3 -m moteus.moteus_tool --target 1 --calibrate