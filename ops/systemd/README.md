# Running Infernum under systemd

Infernum is a long-lived server. If something depends on its embeddings, it needs
supervision — otherwise it dies quietly and the dependent service degrades with no
obvious cause. This directory holds the unit files for that.

## Install (user unit)

```sh
install -m 0755 target/release/infernum ~/.local/bin/infernum
mkdir -p ~/.config/systemd/user ~/.local/share/infernum/logs

cp ops/systemd/infernum.service ops/systemd/infernum-alert.service ~/.config/systemd/user/
# edit the three CHANGE ME paths in infernum.service

systemctl --user daemon-reload
systemctl --user enable --now infernum.service
```

Verify it supervises, rather than assuming it does:

```sh
systemctl --user is-active infernum.service
kill -9 "$(systemctl --user show infernum.service -p MainPID --value)"
sleep 5
systemctl --user is-active infernum.service    # active again, new MainPID
```

A service that restarts but cannot serve is worthless, so check the endpoint too,
not just the process:

```sh
curl -s localhost:11434/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text-v1.5","input":"hello"}' | head -c 120
```

## Two choices worth understanding before you copy this

### Do not point `ExecStart` at `target/release`

Install the binary somewhere stable first. A unit that executes out of a build
directory inside a git checkout runs whatever that checkout last compiled, on
whatever branch it was on — and every resulting failure looks like something
other than a stale binary. Record which commit you installed from; upgrading
should be a deliberate act (rebuild, re-install, `systemctl restart`), never a
side effect of having built something.

### `Restart=always`, not `on-failure`

A clean exit nobody asked for is precisely the outage supervision exists to
prevent, and `on-failure` would let it stand. A deliberate `systemctl stop` is a
state change rather than an exit, so it is never restarted.

## System unit instead

Drop the files in `/etc/systemd/system/`, add `User=` and `Group=`, change
`WantedBy=default.target` to `multi-user.target`, and use `systemctl` without
`--user`.

Note the difference in when they start: **a user unit runs while that user has a
session.** Without lingering enabled (`loginctl enable-linger <user>`) it starts at
login rather than at boot, which for a workstation is usually what you want and for
a server is usually not.

## Alerting

`infernum-alert.service` writes a journal entry and nothing else. That is a
starting point, not a finished alerting story — replace its `ExecStart` with
something that reaches a human.

Note that `OnFailure` fires on **every** unclean exit, not only when the crash-loop
guard trips, because a restarting unit still passes through `failed` on its way
back up. That is usually what you want; if it is too noisy, raise
`StartLimitBurst` or drop `OnFailure` and watch the journal instead.
