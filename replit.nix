{ pkgs }: {
  deps = [
    pkgs.python311Full
    pkgs.postgresql
    pkgs.terraform
  ];
}
